#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
32_optimizacion.py

Registro iterativo de pose CAD↔imagen con:
 - SUSAN (bordes)
 - HPR   (oclusiones)
 - CLP   vectorizado (Closest Line Projection, Sec 3.1.7)
 - M-est Huber (pesos)
 - Q_xy centrada vectorizada
 - ΔR,Δt por SVD/polar
 - Control de tz
 - Escape de mínimos locales con jitter
 - Textura de imagen en plano de fondo (UV corregidas)
 - Visualización y animación Open3D GUI
"""

import os
import math
import random
import threading
import logging

import numpy as np
import cv2
import open3d as o3d
from collections import defaultdict

# ----------------------------------------------------------------------------
# Configuración de logging
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Parámetros de control
# ----------------------------------------------------------------------------
ErrMax           = 0.01      # Umbral PErr
Kmax_init        = 112       # Iteraciones inicialización
Kmax_seg         = 28        # Iteraciones seguimiento
delta_sigma      = 1.483     # Escala robusta
jitter_thresh    = 5         # Iter sin mejora para aplicar jitter
rot_jitter_scale = 0.1       # Radianes para jitter
trans_jitter_scale = 0.02    # Unidades modelo para jitter

state = {
    "K": 0,
    "mode": "Inicial",
    "PErr": np.inf,
    "bestPErr": np.inf,
    "no_improve": 0
}


# ----------------------------------------------------------------------------
# 1) SUSAN Edge Detector
# ----------------------------------------------------------------------------
def susan_edge_detector(gray, t=20, g=28):
    h, w = gray.shape
    out = np.zeros_like(gray, dtype=np.uint8)
    mask = [(-3,0),(-2,-2),(-2,-1),(-2,0),(-2,1),(-2,2),
            (-1,-2),(-1,-1),(-1,0),(-1,1),(-1,2),
            (0,-3),(0,-2),(0,-1),(0,0),(0,1),(0,2),(0,3),
            (1,-2),(1,-1),(1,0),(1,1),(1,2),
            (2,-2),(2,-1),(2,0),(2,1),(2,2),(3,0)]
    for y in range(3, h-3):
        for x in range(3, w-3):
            c = int(gray[y, x])
            area = sum(abs(int(gray[y+dy, x+dx]) - c) < t for dy, dx in mask)
            if area < g:
                out[y, x] = min(255, int((255/g)*(g-area)) + 100)
    return out


# ----------------------------------------------------------------------------
# 2) Build lists from STL
# ----------------------------------------------------------------------------
def build_lists_from_stl(stl_path, interp_spacing=0.5,
                         simplify_triangles=10000, max_total_points=None):
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"No encontré {stl_path}")
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_triangle_normals()
    mesh = mesh.simplify_quadric_decimation(
        target_number_of_triangles=simplify_triangles)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_triangle_normals()

    verts     = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles, dtype=int)
    tri_norms = np.asarray(mesh.triangle_normals)

    V = verts.copy()
    edges = set()
    for tri in triangles:
        for i in range(3):
            a, b = sorted((tri[i], tri[(i+1)%3]))
            edges.add((a, b))
    E = np.array(sorted(edges), dtype=int)

    F3 = []
    eps = 1e-8
    for fid, tri in enumerate(triangles):
        n = tri_norms[fid].astype(float)
        rn = np.linalg.norm(n)
        n = n/rn if rn>eps else np.array([0,0,1], float)
        F3.append([fid, 0x1FF, *n.tolist()])
        for i in range(3):
            ai, bi = tri[i], tri[(i+1)%3]
            perp = np.cross(n, verts[bi]-verts[ai])
            rp = np.linalg.norm(perp)
            perp = perp/rp if rp>eps else np.array([0,0,1], float)
            F3.append([fid, int(ai), *perp.tolist()])
    F3 = np.array(F3, object)

    P4 = []
    for eid, (i,j) in enumerate(E):
        p1, p2 = verts[i], verts[j]
        d = np.linalg.norm(p2-p1)
        segs = max(int(math.ceil(d/interp_spacing)), 1)
        for k in range(segs+1):
            t = k/segs
            P4.append([eid, *(((1-t)*p1 + t*p2).tolist())])
    if max_total_points and len(P4)>max_total_points:
        P4 = random.sample(P4, max_total_points)
    P4 = np.array(P4, float)

    return mesh, V, E, F3, P4


# ----------------------------------------------------------------------------
# 3) Visibilidad HPR
# ----------------------------------------------------------------------------
def calcular_visibilidad_hpr(P4, K=None, img_shape=None):
    pts = P4[:,1:]
    pc_all = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    camera = np.zeros(3)
    radius = np.linalg.norm(pts,axis=1).max()*1.1
    pc_vis, idx = pc_all.hidden_point_removal(camera, radius)
    vis = pts[idx]

    if K is not None and img_shape:
        h, w = img_shape
        proj = (K @ vis.T).T
        mask_z = proj[:,2] > 1e-6
        proj = proj[mask_z]; vis = vis[mask_z]
        uv = proj[:,:2] / proj[:,2:3]
        m = (uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        vis = vis[m]

    return vis


# ----------------------------------------------------------------------------
# 3.1) Emparejamiento CLP vectorizado
# ----------------------------------------------------------------------------
def emparejar_clp(Pvis, rays):
    if len(Pvis)==0:
        return np.zeros((0,3)), np.zeros((0,3))
    dots = Pvis @ rays.T
    idx  = np.argmax(dots, axis=1)
    d    = dots[np.arange(len(Pvis)), idx]
    r    = rays[idx]
    ParM = Pvis
    ParI = r * d[:,None]
    return ParM, ParI


# ----------------------------------------------------------------------------
# 3.2) Cálculo de pesos Huber
# ----------------------------------------------------------------------------
def calcular_pesos(ParM, ParI, k_iter):
    dif = ParI - ParM
    sq  = np.sum(dif*dif, axis=1)
    ErrMP = math.sqrt(np.mean(sq))
    a = 1.0 if k_iter<5 else 2.0
    sigma2 = a * ErrMP * delta_sigma
    w = np.where(sq<sigma2, 1.0, sigma2/np.maximum(sq,1e-8))
    return w, ErrMP


# ----------------------------------------------------------------------------
# 3.3) Construir Q_xy centrada vectorizado
# ----------------------------------------------------------------------------
def construir_Qxy(ParM, ParI, w):
    wsum = w.sum()
    muM  = (w[:,None]*ParM).sum(axis=0)/wsum
    muI  = (w[:,None]*ParI).sum(axis=0)/wsum
    dM   = ParM - muM
    dI   = ParI - muI
    Q    = (dM * w[:,None]).T @ dI
    return Q, muM, muI


# ----------------------------------------------------------------------------
# 3.4) Estimar R,t por SVD/polar
# ----------------------------------------------------------------------------
def estimar_R_t(Q, muM, muI):
    U,_,VT = np.linalg.svd(Q)
    R_inc  = U @ VT
    if np.linalg.det(R_inc)<0:
        U[:,-1]*=-1
        R_inc = U @ VT
    t_inc = muM - R_inc @ muI
    return R_inc, t_inc


# ----------------------------------------------------------------------------
# 3.5) Control de tz
# ----------------------------------------------------------------------------
def controlar_tz(tz, PErr):
    if   PErr>=0.05:
        kt = 1.25
    elif PErr<=-0.05:
        kt = 0.75
    else:
        kt = 1 + 0.5*PErr
    return tz * kt


# ----------------------------------------------------------------------------
# 4) LineSet aristas
# ----------------------------------------------------------------------------
def construir_lineset(P4):
    ptsln, lnls, off = [], [], 0
    by_e = defaultdict(list)
    for eid,x,y,z in P4:
        by_e[int(eid)].append([x,y,z])
    for pts in by_e.values():
        for i in range(len(pts)-1):
            ptsln += [pts[i], pts[i+1]]
            lnls.append([off, off+1])
            off += 2
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(ptsln)),
        lines =o3d.utility.Vector2iVector(np.asarray(lnls))
    )
    ls.paint_uniform_color([1,0,1])
    return ls


# ----------------------------------------------------------------------------
# 5) Callback iterativo con jitter
# ----------------------------------------------------------------------------
def actualizar_visibilidad_dinamica(
    R_tot, t_tot, V0, P40,
    R_flip, K, img_shape, rays,
    scene, win, vis_pc
):
    state["K"] += 1
    k = state["K"]

    # 1) Aplicar pose
    V  = (R_tot @ V0.T).T + t_tot
    P4 = P40.copy()
    P4[:,1:] = (R_tot @ P4[:,1:].T).T + t_tot
    np.save("Lista1_vertices.npy", V)
    np.save("Lista4_interpolados.npy", P4)

    # 2) Visibilidad
    Pvis = calcular_visibilidad_hpr(P4, K, img_shape)

    # 3) CLP
    ParM, ParI = emparejar_clp(Pvis, rays)

    # 4) Pesos y ErrMP
    w, ErrMP = calcular_pesos(ParM, ParI, k)
    PErr = ErrMP / max(np.linalg.norm(V.mean(axis=0)), 1e-8)
    state["PErr"] = PErr
    logger.info(f"Iter {k} [{state['mode']}]: ErrMP={ErrMP:.4f}, PErr={PErr:.4f}")

    # 5) Escape de mínimos locales con jitter
    if PErr < state["bestPErr"]:
        state["bestPErr"] = PErr
        state["no_improve"] = 0
    else:
        state["no_improve"] += 1

    jit_R, jit_t = None, None
    if state["no_improve"] > jitter_thresh:
        rv = np.random.normal(scale=rot_jitter_scale, size=3)
        R_pert = o3d.geometry.get_rotation_matrix_from_axis_angle(rv)
        t_pert = np.random.normal(scale=trans_jitter_scale, size=3)
        logger.info(f"🔀 Jitter en iter {k}: rot={rv}, trans={t_pert}")
        jit_R, jit_t = R_pert, t_pert
        state["no_improve"] = 0

    # 6) Q_xy y ΔR,Δt
    Q, muM, muI = construir_Qxy(ParM, ParI, w)
    R_inc, t_inc = estimar_R_t(Q, muM, muI)

    if jit_R is not None:
        R_inc = jit_R @ R_inc
        t_inc = jit_R @ t_inc + jit_t

    # 7) Control tz
    t_inc[2] = controlar_tz(t_inc[2], PErr)

    # 8) Acumular pose
    R_tot_new = R_inc @ R_tot
    t_tot_new = R_inc @ t_tot + t_inc

    # 9) Visualización
    Pvis_n = calcular_visibilidad_hpr(P4, K, img_shape)
    vis_pc.points = o3d.utility.Vector3dVector(Pvis_n)
    vis_pc.paint_uniform_color([0,0,1])
    scene.remove_geometry("Puntos_Visibles")
    scene.add_geometry("Puntos_Visibles", vis_pc, o3d.visualization.rendering.MaterialRecord())

    ls = construir_lineset(P4)
    scene.remove_geometry("Aristas_Interpoladas")
    scene.add_geometry("Aristas_Interpoladas", ls, o3d.visualization.rendering.MaterialRecord())

    # 10) Cambio de modo y parada
    if state["mode"]=="Inicial" and (PErr<=ErrMax or k>=Kmax_init):
        state["mode"]="Seguimiento"
        logger.info("➡️ Pasa a SEGUIMIENTO")
    if state["mode"]=="Seguimiento" and (PErr<=ErrMax or k>=Kmax_seg):
        logger.info("✅ Convergencia en iter %d", k)
        return

    # 11) Próxima iteración
    threading.Timer(1.5, lambda:
        o3d.visualization.gui.Application.instance.post_to_main_thread(
            win,
            lambda: actualizar_visibilidad_dinamica(
                R_tot_new, t_tot_new,
                V0, P40,
                R_flip, K, img_shape, rays,
                scene, win, vis_pc
            )
        )
    ).start()


# ----------------------------------------------------------------------------
# 6) Main
# ----------------------------------------------------------------------------
def main():
    STL, CALIB, IMG = "Benchy.stl", "camera_calibration.npz", "foto.jpg"
    mesh, V0, E, F3, P40 = build_lists_from_stl(
        STL, interp_spacing=0.5,
        simplify_triangles=10000,
        max_total_points=50000
    )
    scale = 0.02
    V0   *= scale
    P40[:,1:] *= scale

    # Cámara
    with np.load(CALIB) as D:
        K = D["camera_matrix"]
    K_inv  = np.linalg.inv(K)
    R_flip = np.diag([-1,-1,1])

    img      = cv2.imread(IMG)
    if img is None:
        raise FileNotFoundError(IMG)
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w     = img.shape[:2]

    # SUSAN + rays
    edges = susan_edge_detector(img_gray)
    ys, xs = np.where(edges>0)
    pts2d  = np.stack([xs, ys], axis=-1).astype(np.float32)
    if len(pts2d)>400:
        pts2d = pts2d[np.random.choice(len(pts2d),400,replace=False)]
    rays = (K_inv @ np.hstack([pts2d, np.ones((len(pts2d),1))]).T).T
    rays = (R_flip @ rays.T).T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)

    # GUI setup
    app    = o3d.visualization.gui.Application.instance; app.initialize()
    win    = app.create_window("📷 Registro Iterativo CAD↔Imagen", 1024, 768)
    widget = o3d.visualization.gui.SceneWidget()
    widget.scene = o3d.visualization.rendering.Open3DScene(win.renderer)

    # ─── Plano texturizado sin espeje ───
    corn2d = np.array([[0,0],[w,0],[w,h],[0,h]], float)
    corn3d = []
    for x,y in corn2d:
        v = K_inv @ np.array([x,y,1.])
        v /= v[2]
        corn3d.append(R_flip @ v)
    corn3d = np.vstack(corn3d)

    bg = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(corn3d),
        triangles=o3d.utility.Vector3iVector([[0,1,2],[0,2,3]])
    )
    # UV invertidas solo en v:
    bg.triangle_uvs = o3d.utility.Vector2dVector([
        [0.0, 1.0],  # vert 0 → top-left
        [1.0, 1.0],  # vert 1 → top-right
        [1.0, 0.0],  # vert 2 → bottom-right

        [0.0, 1.0],  # vert 0
        [1.0, 0.0],  # vert 2
        [0.0, 0.0],  # vert 3 → bottom-left
    ])
    bg.textures = [o3d.io.read_image(IMG)]
    bg.compute_vertex_normals()

    mat_bg         = o3d.visualization.rendering.MaterialRecord()
    mat_bg.shader  = "defaultUnlit"
    mat_bg.albedo_img = o3d.io.read_image(IMG)
    widget.scene.add_geometry("Imagen", bg, mat_bg)

    # Frustum y ejes
    fr_pts = np.vstack([corn3d, [[0,0,0]]])
    fr_ls  = [[0,1],[1,2],[2,3],[3,0],[0,4],[1,4],[2,4],[3,4]]
    fru    = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(fr_pts),
        lines =o3d.utility.Vector2iVector(fr_ls)
    ); fru.paint_uniform_color([1,0,0])
    axis   = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    widget.scene.add_geometry("Frustum", fru,  o3d.visualization.rendering.MaterialRecord())
    widget.scene.add_geometry("Ejes",    axis, o3d.visualization.rendering.MaterialRecord())

    # SUSAN puntos y rayos
    pc_susan = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rays*2.0))
    pc_susan.paint_uniform_color([0,1,0])
    ls_rayos = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.zeros_like(rays*2.0))),
        pc_susan,
        list(zip(range(len(rays)), range(len(rays))))
    ); ls_rayos.paint_uniform_color([0.2,0.6,1.0])
    widget.scene.add_geometry("SUSAN_Points", pc_susan, o3d.visualization.rendering.MaterialRecord())
    widget.scene.add_geometry("Rayos",        ls_rayos, o3d.visualization.rendering.MaterialRecord())

    # Pose inicial
    R0 = o3d.geometry.get_rotation_matrix_from_axis_angle(
        [-math.pi/3, -math.pi, -math.pi/2]
    )
    t0 = np.array([0.,0.,2.5])
    widget.setup_camera(60.0, widget.scene.bounding_box, [0,0,0])
    win.add_child(widget)

    # Visual inicial
    vis0 = calcular_visibilidad_hpr(P40, K, (h,w))
    vis_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vis0))
    vis_pc.paint_uniform_color([0,0,1])
    widget.scene.add_geometry("Puntos_Visibles", vis_pc, o3d.visualization.rendering.MaterialRecord())
    widget.scene.add_geometry("Aristas_Interpoladas", construir_lineset(P40),
                              o3d.visualization.rendering.MaterialRecord())

    # Inicio iteraciones
    threading.Timer(1.0, lambda:
        actualizar_visibilidad_dinamica(
            R0, t0, V0, P40,
            R_flip, K, (h,w), rays,
            widget.scene, win, vis_pc
        )
    ).start()

    app.run()


if __name__ == "__main__":
    main()
