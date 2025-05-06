# 📦 Código completo: Visualización con puntos visibles (Lista5) (Cuadro 3.5) – CON SCALE Y Hidden Point Removal
import open3d as o3d
import numpy as np
import cv2
from collections import defaultdict
import math, random, os

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

def build_lists_from_stl(stl_path,
                         interp_spacing=0.5,
                         simplify_triangles=10000,
                         max_total_points=None):
    # 1) Leer y limpiar
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_triangle_normals()
    # 2) Decimar a un nº razonable de caras
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=simplify_triangles)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_triangle_normals()

    verts     = np.asarray(mesh.vertices)        # (nV,3)
    triangles = np.asarray(mesh.triangles,   int) # (nF,3)
    tri_norms = np.asarray(mesh.triangle_normals) # (nF,3)

    # === Lista1: vértices ===
    V = verts.copy()

    # === Lista2: aristas únicas ===
    edges = set()
    for tri in triangles:
        for i in range(3):
            a, b = sorted((tri[i], tri[(i+1)%3]))
            edges.add((a,b))
    E = np.array(sorted(edges), dtype=int)

    # === Lista3: normales + perpendiculares (no usado en HPR) ===
    F3 = []
    eps = 1e-8
    for fid, tri in enumerate(triangles):
        n = tri_norms[fid].astype(float)
        norm_n = np.linalg.norm(n)
        if norm_n>eps: n /= norm_n
        else:          n = np.array([0.,0.,1.])
        F3.append([fid, 0x1FF, *n.tolist()])
        for i in range(3):
            ai, bi = tri[i], tri[(i+1)%3]
            edge_vec = verts[bi] - verts[ai]
            perp = np.cross(n, edge_vec)
            norm_p = np.linalg.norm(perp)
            if norm_p>eps: perp /= norm_p
            else:           perp = np.array([0.,0.,1.])
            F3.append([fid, int(ai), *perp.tolist()])
    F3 = np.array(F3, dtype=object)

    # === Lista4: puntos interpolados en aristas ===
    P4 = []
    for eid, (i,j) in enumerate(E):
        p1, p2 = verts[i], verts[j]
        d = np.linalg.norm(p2-p1)
        segs = max(int(math.ceil(d/interp_spacing)), 1)
        for k in range(segs+1):
            t = k / segs
            P4.append([eid, *(((1-t)*p1 + t*p2).tolist())])
    if max_total_points and len(P4) > max_total_points:
        P4 = random.sample(P4, max_total_points)
    P4 = np.array(P4, dtype=float)

    return mesh, V, E, F3, P4

# ── Parámetros ────────────────────────────────────────────────────────────
STL = "Benchy.stl"
if not os.path.exists(STL):
    raise FileNotFoundError(f"No encontré {STL}")

# ── 0) Generar listas + malla decimada ───────────────────────────────────
mesh_dec, V, E, F3, P4 = build_lists_from_stl(
    STL,
    interp_spacing=0.5,
    simplify_triangles=10000,
    max_total_points=50000
)

# ── 0.5) Aplicar scale factor a listas ──────────────────────────────────
scale = 0.02
V       *= scale
P4[:,1:] *= scale
# (F3 contiene vectores unitarios: no se escala)

# ── 1) Intrínsecos + carga de imagen ────────────────────────────────────
with np.load("camera_calibration.npz") as D:
    K = D["camera_matrix"]
K_inv  = np.linalg.inv(K)
R_flip = np.diag([-1,-1,1])

img      = cv2.imread("foto.jpg")
img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w     = img.shape[:2]

# ── 2) SUSAN + rayos ────────────────────────────────────────────────────
edges2d = susan_edge_detector(img_gray)
ys, xs  = np.where(edges2d > 0)
pts2d   = np.stack([xs, ys], axis=-1).astype(np.float32)
if len(pts2d) > 400:
    pts2d = pts2d[np.random.choice(len(pts2d), 400, replace=False)]
rays = (K_inv @ np.hstack([pts2d, np.ones((len(pts2d),1))]).T).T
rays = (R_flip @ rays.T).T
rays /= np.linalg.norm(rays, axis=1, keepdims=True)
pts3d = rays * 2.0

pc_susan = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts3d))
pc_susan.paint_uniform_color([0,1,0])
rayos_ls = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
    o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.zeros((len(pts3d),3)))),
    pc_susan,
    list(zip(range(len(pts3d)), range(len(pts3d))))
)
rayos_ls.paint_uniform_color([0.2,0.6,1.0])

# ── 3) Plano de fondo + frustum + ejes ───────────────────────────────────
corn2d = np.array([[0,0],[w,0],[w,h],[0,h]], float)
corn3d = []
for x,y in corn2d:
    v = K_inv @ np.array([x,y,1.])
    v /= v[2]
    corn3d.append(R_flip @ v)
corn3d = np.array(corn3d)

bg = o3d.geometry.TriangleMesh(
    vertices=o3d.utility.Vector3dVector(corn3d),
    triangles=o3d.utility.Vector3iVector([[0,1,2],[0,2,3]])
)
bg.triangle_uvs = o3d.utility.Vector2dVector([
    [1,0],[0,0],[0,1],
    [1,0],[0,1],[1,1]
])
cv2.imwrite("temp_image.jpg", cv2.flip(img_rgb, -1))
bg.textures = [o3d.io.read_image("temp_image.jpg")]
bg.compute_vertex_normals()

fr_pts = np.vstack([corn3d, [[0,0,0]]])
fr_ls  = [[0,1],[1,2],[2,3],[3,0],[0,4],[1,4],[2,4],[3,4]]
fru = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(fr_pts),
    lines=o3d.utility.Vector2iVector(fr_ls)
)
fru.paint_uniform_color([1,0,0])
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

# ── 4) Occlusion removal con Hidden Point Removal ───────────────────────
# Construir PointCloud de todos los puntos interpolados (tras pose, ver siguiente paso)
# Primero aplicamos la pose a V y P4
R0 = o3d.geometry.get_rotation_matrix_from_axis_angle([-np.pi/3, -np.pi, -np.pi/2])
t0 = np.array([0.,0.,2.5])
V       = (R0 @ V.T).T + t0
P4[:,1:] = (R0 @ P4[:,1:].T).T + t0

# Construir la nube
pc_all = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P4[:,1:]))

# Cámara en el origen del sistema
camera = np.array([0.,0.,0.], dtype=float)
# Radio algo mayor que la distancia máxima de los puntos
radius = np.linalg.norm(P4[:,1:], axis=1).max() * 1.1

# Ejecutar el filtro HPR
pc_vis, idx_visible = pc_all.hidden_point_removal(camera, radius)

# Extraer los puntos visibles
P5 = P4[idx_visible, 1:]   # (n_visibles, 3)
np.save("Lista5_visibles.npy", P5)

# ── 5) Preparar LineSet de interpolados para visualizar aristas ─────────
ptsln, lnls, off = [], [], 0
# reagrupar P4 por arista
pts_by_e = defaultdict(list)
for eid, x, y, z in P4:
    pts_by_e[int(eid)].append([x,y,z])
for pts in pts_by_e.values():
    for i in range(len(pts)-1):
        ptsln += [pts[i], pts[i+1]]
        lnls.append([off, off+1])
        off += 2
interpol = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(ptsln),
    lines=o3d.utility.Vector2iVector(lnls)
)
interpol.paint_uniform_color([1,0,1])

vis_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P5))
vis_pc.paint_uniform_color([1,1,0])

# ── 6) GUI ───────────────────────────────────────────────────────────────
app = o3d.visualization.gui.Application.instance; app.initialize()
win   = app.create_window("📷 CAD + SUSAN + Lista5", 1024, 768)
scene = o3d.visualization.gui.SceneWidget()
scene.scene = o3d.visualization.rendering.Open3DScene(win.renderer)

mat_bg = o3d.visualization.rendering.MaterialRecord()
mat_bg.shader     = "defaultUnlit"
mat_bg.albedo_img = o3d.io.read_image("temp_image.jpg")

scene.scene.add_geometry("Imagen",               bg,       mat_bg)
scene.scene.add_geometry("Frustum",              fru,      o3d.visualization.rendering.MaterialRecord())
scene.scene.add_geometry("Ejes",                 axis,     o3d.visualization.rendering.MaterialRecord())
scene.scene.add_geometry("SUSAN_Points",         pc_susan,o3d.visualization.rendering.MaterialRecord())
scene.scene.add_geometry("Rayos",                rayos_ls, o3d.visualization.rendering.MaterialRecord())
scene.scene.add_geometry("Aristas_Interpoladas", interpol,o3d.visualization.rendering.MaterialRecord())
scene.scene.add_geometry("Puntos_Visibles",      vis_pc,   o3d.visualization.rendering.MaterialRecord())

print(f"🔍 Porcentaje de puntos visibles: {100 * len(P5) / len(P4):.2f}%")
scene.setup_camera(60.0, scene.scene.bounding_box, [0,0,0])
win.add_child(scene)
app.run()
