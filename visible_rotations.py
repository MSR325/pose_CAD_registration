import open3d as o3d
import numpy as np
import cv2
from collections import defaultdict
import math, random, os, threading

# === SUSAN detector ===
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
            c = int(gray[y,x])
            area = sum(1 for dy,dx in mask if abs(int(gray[y+dy,x+dx]) - c) < t)
            if area < g:
                out[y,x] = min(255, int((255/g)*(g-area)) + 100)
    return out

# === STL → Listas ===
def build_lists_from_stl(stl_path, interp_spacing=0.5, simplify_triangles=10000, max_total_points=None):
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_triangle_normals()
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=simplify_triangles)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_triangle_normals()

    verts = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles, int)
    tri_norms = np.asarray(mesh.triangle_normals)

    V = verts.copy()
    edges = set()
    for tri in triangles:
        for i in range(3):
            a, b = sorted((tri[i], tri[(i+1)%3]))
            edges.add((a,b))
    E = np.array(sorted(edges), dtype=int)

    F3 = []
    eps = 1e-8
    for fid, tri in enumerate(triangles):
        n = tri_norms[fid].astype(float)
        norm_n = np.linalg.norm(n)
        n = n / norm_n if norm_n > eps else np.array([0.,0.,1.])
        F3.append([fid, 0x1FF, *n.tolist()])
        for i in range(3):
            ai, bi = tri[i], tri[(i+1)%3]
            edge_vec = verts[bi] - verts[ai]
            perp = np.cross(n, edge_vec)
            norm_p = np.linalg.norm(perp)
            perp = perp / norm_p if norm_p > eps else np.array([0.,0.,1.])
            F3.append([fid, int(ai), *perp.tolist()])
    F3 = np.array(F3, dtype=object)

    P4 = []
    for eid, (i,j) in enumerate(E):
        p1, p2 = verts[i], verts[j]
        d = np.linalg.norm(p2-p1)
        segs = max(int(math.ceil(d/interp_spacing)), 1)
        for k in range(segs+1):
            t = k / segs
            P4.append([eid, *((1-t)*p1 + t*p2)])
    if max_total_points and len(P4) > max_total_points:
        P4 = random.sample(P4, max_total_points)
    P4 = np.array(P4, dtype=float)

    return mesh, V, E, F3, P4

# === Carga de imagen, cámara y STL ===
STL = "Benchy.stl"
mesh_dec, V, E, F3, P4 = build_lists_from_stl(STL, interp_spacing=0.5, max_total_points=50000)
scale = 0.02
V *= scale
P4[:,1:] *= scale

with np.load("camera_calibration.npz") as D:
    K = D["camera_matrix"]
K_inv = np.linalg.inv(K)
R_flip = np.diag([-1,-1,1])

img = cv2.imread("foto.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = img.shape[:2]

# === SUSAN + Rayos ===
edges2d = susan_edge_detector(img_gray)
ys, xs = np.where(edges2d > 0)
pts2d = np.stack([xs, ys], axis=-1).astype(np.float32)
if len(pts2d) > 400:
    pts2d = pts2d[np.random.choice(len(pts2d), 400, replace=False)]
rays = (K_inv @ np.hstack([pts2d, np.ones((len(pts2d),1))]).T).T
rays = (R_flip @ rays.T).T
rays /= np.linalg.norm(rays, axis=1, keepdims=True)
pts3d = rays * 2.0

pc_susan = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts3d))
pc_susan.paint_uniform_color([0,1,0])
rayos_ls = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
    o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.zeros_like(pts3d))),
    pc_susan,
    list(zip(range(len(pts3d)), range(len(pts3d))))
)
rayos_ls.paint_uniform_color([0.2,0.6,1.0])

# === Imagen y frustum ===
corn2d = np.array([[0,0],[w,0],[w,h],[0,h]], float)
corn3d = [R_flip @ (K_inv @ np.array([x,y,1.]) / (K_inv @ np.array([x,y,1.]))[2]) for x,y in corn2d]
bg = o3d.geometry.TriangleMesh(
    vertices=o3d.utility.Vector3dVector(np.array(corn3d)),
    triangles=o3d.utility.Vector3iVector([[0,1,2],[0,2,3]])
)
bg.triangle_uvs = o3d.utility.Vector2dVector([
    [1,0],[0,0],[0,1],[1,0],[0,1],[1,1]
])
cv2.imwrite("temp_image.jpg", cv2.flip(img_rgb, -1))
bg.textures = [o3d.io.read_image("temp_image.jpg")]
bg.compute_vertex_normals()

fr_pts = np.vstack([corn3d, [[0,0,0]]])
fr_ls  = [[0,1],[1,2],[2,3],[3,0],[0,4],[1,4],[2,4],[3,4]]
fru = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(fr_pts), lines=o3d.utility.Vector2iVector(fr_ls))
fru.paint_uniform_color([1,0,0])
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

# === Estado inicial ===
R0 = o3d.geometry.get_rotation_matrix_from_axis_angle([-np.pi/3, -np.pi, -np.pi/2])
t0 = np.array([0.,0.,2.5])
V = (R0 @ V.T).T + t0
P4[:,1:] = (R0 @ P4[:,1:].T).T + t0

# Guardar listas transformadas (paso 0)
np.save("Lista1_vertices.npy", V)
np.save("Lista4_interpolados.npy", P4)

# === Visualizador ===
app = o3d.visualization.gui.Application.instance; app.initialize()
win = app.create_window("📷 CAD + SUSAN + Transformaciones", 1024, 768)
scene = o3d.visualization.gui.SceneWidget()
scene.scene = o3d.visualization.rendering.Open3DScene(win.renderer)

mat_bg = o3d.visualization.rendering.MaterialRecord()
mat_bg.shader = "defaultUnlit"
mat_bg.albedo_img = o3d.io.read_image("temp_image.jpg")

scene.scene.add_geometry("Imagen", bg, mat_bg)
scene.scene.add_geometry("Frustum", fru, o3d.visualization.rendering.MaterialRecord())
scene.scene.add_geometry("Ejes", axis, o3d.visualization.rendering.MaterialRecord())
scene.scene.add_geometry("SUSAN_Points", pc_susan, o3d.visualization.rendering.MaterialRecord())
scene.scene.add_geometry("Rayos", rayos_ls, o3d.visualization.rendering.MaterialRecord())

# === Inicialización LineSet + visibles ===
def construir_lineset(P4):
    ptsln, lnls, off = [], [], 0
    pts_by_e = defaultdict(list)
    for eid, x, y, z in P4:
        pts_by_e[int(eid)].append([x,y,z])
    for pts in pts_by_e.values():
        for i in range(len(pts)-1):
            ptsln += [pts[i], pts[i+1]]
            lnls.append([off, off+1])
            off += 2
    lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(ptsln),
        lines=o3d.utility.Vector2iVector(lnls)
    )
    lineset.paint_uniform_color([1,0,1])
    return lineset

# === Visibilidad por rasterización Z-buffer con cámara calibrada ===
def calcular_visibilidad_zbuffer(P4, K, img_shape):
    h, w = img_shape
    puntos_3d = P4[:, 1:]
    puntos_cam = puntos_3d.copy()  # Asumimos ya están en marco cámara

    proy = (K @ puntos_cam.T).T
    proy = proy[:, :2] / proy[:, 2:3]

    x_pix = np.round(proy[:, 0]).astype(int)
    y_pix = np.round(proy[:, 1]).astype(int)

    mask = (x_pix >= 0) & (x_pix < w) & (y_pix >= 0) & (y_pix < h)
    x_pix, y_pix = x_pix[mask], y_pix[mask]
    z_vals = puntos_cam[mask][:, 2]
    idxs = np.where(mask)[0]

    z_buffer = np.full((h, w), np.inf, dtype=np.float32)
    vis_idxs = []

    for i, x, y, z in zip(idxs, x_pix, y_pix, z_vals):
        if z < z_buffer[y, x]:
            z_buffer[y, x] = z
            vis_idxs.append(i)

    return P4[vis_idxs, 1:]


interpol = construir_lineset(P4)
P5 = calcular_visibilidad_zbuffer(P4, K, (h, w))
vis_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P5))
vis_pc.paint_uniform_color([1,1,0])

scene.scene.add_geometry("Aristas_Interpoladas", interpol, o3d.visualization.rendering.MaterialRecord())
scene.scene.add_geometry("Puntos_Visibles", vis_pc, o3d.visualization.rendering.MaterialRecord())
scene.setup_camera(60.0, scene.scene.bounding_box, [0,0,0])
win.add_child(scene)

# === Animación incremental ===
num_steps = 5
step_idx = 0
V_orig = V.copy()
P4_orig = P4.copy()

def aplicar_transformacion_incremental(V_in, P4_in, step):
    angle = (step + 1) * (np.pi / 12)
    rotvec = np.array([angle, angle/2, angle/3])
    Rk = o3d.geometry.get_rotation_matrix_from_axis_angle(rotvec)
    tk = np.array([0.05 * step, -0.03 * step, 0.07 * step])
    V_new = (Rk @ V_in.T).T + tk
    P4_tmp = P4_in.copy()
    P4_tmp[:,1:] = (Rk @ P4_tmp[:,1:].T).T + tk
    return Rk, tk, V_new, P4_tmp

def actualizar_visibilidad_dinamica(step):
    global V, P4, interpol, vis_pc

    Rk, tk, Vn, P4n = aplicar_transformacion_incremental(V_orig, P4_orig, step)
    V, P4 = Vn, P4n

    # Actualizar archivos
    np.save("Lista1_vertices.npy", V)
    np.save("Lista4_interpolados.npy", P4)

    # Recalcular listas
    P5n = calcular_visibilidad_zbuffer(P4, K, (h, w))
    vis_pc.points = o3d.utility.Vector3dVector(P5n)
    vis_pc.paint_uniform_color([1,1,0])
    scene.scene.remove_geometry("Puntos_Visibles")
    scene.scene.add_geometry("Puntos_Visibles", vis_pc, o3d.visualization.rendering.MaterialRecord())

    # Recalcular LineSet
    interpol = construir_lineset(P4)
    scene.scene.remove_geometry("Aristas_Interpoladas")
    scene.scene.add_geometry("Aristas_Interpoladas", interpol, o3d.visualization.rendering.MaterialRecord())

    print(f"\n🌀 Paso {step+1}/{num_steps}")
    print("🔁 Rotación incremental:", np.round(Rk, 3))
    print("🚀 Traslación aplicada:", np.round(tk, 3))
    print(f"🔍 Porcentaje de puntos visibles: {100 * len(P5n) / len(P4):.2f}%")

def loop_transformaciones():
    global step_idx
    if step_idx < num_steps:
        app.post_to_main_thread(win, lambda: actualizar_visibilidad_dinamica(step_idx))
        step_idx += 1
        threading.Timer(1.5, loop_transformaciones).start()

threading.Timer(1.0, loop_transformaciones).start()
app.run()
