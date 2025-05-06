import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as Rscipy
import json

# === Cargar intrínsecos ===
def load_intrinsics(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    fs.release()
    return K

# === Bordes con Canny ===
def detect_image_edges(gray):
    edges = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    pts = np.vstack([cnt[:, 0, :] for cnt in cnts]).astype(np.float32)
    return pts

# === Rayos desde píxeles 2D ===
def compute_rays(K):
    K_inv = np.linalg.inv(K)
    def rays_from(pts2d):
        homog = np.hstack([pts2d, np.ones((len(pts2d), 1))])
        dirs = (K_inv @ homog.T).T
        return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    return rays_from

# === Proyección wireframe ===
def proyectar_mesh_en_imagen(mesh, K, R, t, img, color=(0, 255, 0)):
    img_overlay = img.copy()
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    projected = []

    for v in vertices:
        p = K @ (R @ v + t)
        if p[2] <= 0:
            projected.append(None)
            continue
        p /= p[2]
        projected.append((int(p[0]), int(p[1])))

    for tri in triangles:
        try:
            pts = [projected[i] for i in tri]
            if any(p is None for p in pts):
                continue
            pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_overlay, [pts], isClosed=True, color=color, thickness=1)
        except:
            continue

    return img_overlay

# === CLP híbrido ===
def clp_hybrid(model_cam_pts, img_pts2d, rays_fn, K, k=10):
    P = (K @ model_cam_pts.T).T
    proj2d = P[:, :2] / P[:, 2:3]
    tree = KDTree(proj2d)
    rays = rays_fn(img_pts2d)
    x_list, y_list = [], []
    for i, uv in enumerate(img_pts2d):
        _, idxs = tree.query(uv, k=k)
        best_dist = np.inf
        best_x = best_y = None
        for j in np.atleast_1d(idxs):
            pj = model_cam_pts[j]
            ray = rays[i]
            lam = ray.dot(pj)
            proj3d = ray * lam
            d3 = np.linalg.norm(pj - proj3d)
            if d3 < best_dist:
                best_dist, best_x, best_y = d3, pj, proj3d
        x_list.append(best_x)
        y_list.append(best_y)
    return np.vstack(x_list), np.vstack(y_list)

# === SVD con corrección de reflexión ===
def register_svd(x, y, w=None):
    if w is None:
        w = np.ones(len(x))
    cx = np.average(x, axis=0, weights=w)
    cy = np.average(y, axis=0, weights=w)
    Xc, Yc = x - cx, y - cy
    H = (Xc * w[:, None]).T @ Yc
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cy - R @ cx
    return R, t

# === Huber weights ===
def compute_huber_weights(res):
    δ = max(1e-8, 1.4826 * np.median(res))
    w = np.ones_like(res)
    mask = np.abs(res) > δ
    w[mask] = δ / np.abs(res[mask])
    return w

# === Convergencia ===
def check_convergence(errs, tΔ, RΔ, tol_e, tol_t, tol_r):
    if len(errs) < 2:
        return False
    if np.abs(errs[-1] - errs[-2]) > tol_e:
        return False
    if np.linalg.norm(tΔ) < tol_t:
        angle = np.arccos((np.trace(RΔ) - 1) / 2)
        return abs(angle) < tol_r
    return False

# === ICP-CLP iterativo ===
def iterative_clp_register(mesh, model_pts, img_pts2d, rays_fn, K,
                           R0, t0, gray_img, max_iter=80,
                           tol_e=1e-3, tol_t=1e-4, tol_r=1e-2, k=10):

    R, t = R0.copy(), t0.copy()
    errs = []

    fig, ax = plt.subplots()
    line, = ax.plot([], [], marker='o')
    ax.set_title("Error medio por iteración")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Error")
    ax.grid(True)

    for i in range(max_iter):
        cam_pts = (R @ model_pts.T).T + t
        x, y = clp_hybrid(cam_pts, img_pts2d, rays_fn, K, k)
        res = np.linalg.norm(x - y, axis=1)
        err = res.mean()
        errs.append(err)
        w = compute_huber_weights(res)
        RΔ, tΔ = register_svd(x, y, w)
        R, t = RΔ @ R, RΔ @ t + tΔ

        overlay = proyectar_mesh_en_imagen(mesh, K, R, t, gray_img, color=(0, 255, 0))
        cv2.imshow("Registro CAD sobre imagen", overlay)
        cv2.waitKey(30)

        line.set_ydata(errs)
        line.set_xdata(range(len(errs)))
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.05)

        if check_convergence(errs, tΔ, RΔ, tol_e, tol_t, tol_r):
            break

    cv2.imwrite("pose_estimada_proyeccion_final.png", overlay)
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()
    return R, t, errs

# === Profundidad inicial ===
def estimate_initial_depth(model_pts, img_pts2d, K):
    Dpx = max(img_pts2d[:, 0].ptp(), 1.0)
    D3d = model_pts[:, 0].ptp()
    fx = K[0, 0]
    z0 = fx * D3d / Dpx
    return z0

# === Multistart ===
def multistart(mesh, model_pts, img_pts2d, rays_fn, K, gray_img,
               n_starts=5, **kw):
    best, be = None, np.inf
    Z0 = estimate_initial_depth(model_pts, img_pts2d, K)
    for i in range(n_starts):
        ang = np.random.uniform(-0.1, 0.1, 3)
        R0 = Rscipy.from_rotvec(ang).as_matrix()
        t0 = np.array([0, 0, Z0]) + np.random.normal(scale=0.1 * Z0, size=3)
        Rf, tf, errs = iterative_clp_register(mesh, model_pts, img_pts2d, rays_fn, K, R0, t0, gray_img, **kw)
        if errs[-1] < be:
            be, best = errs[-1], (Rf, tf, errs)
    return best

# === Guardar resultado JSON ===
def save_pose_json(R, t, path="benchy_pose.json"):
    with open(path, "w") as f:
        json.dump({
            "rotation": R.tolist(),
            "translation": t.tolist()
        }, f, indent=2)

# === MAIN ===
if __name__ == "__main__":
    # Entradas
    mesh_path = "Benchy.stl"
    img_path = "benchy_image4.jpg"
    intrinsics_path = "camera_intrinsics.yaml"

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    K = load_intrinsics(intrinsics_path)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pts2d = detect_image_edges(gray)
    rays_fn = compute_rays(K)
    silhouette = mesh.sample_points_uniformly(3000).points
    silhouette = np.asarray(silhouette) - np.mean(np.asarray(silhouette), axis=0)

    R, t, errs = multistart(
        mesh, silhouette, pts2d, rays_fn, K, gray,
        n_starts=5, max_iter=80, tol_e=1e-3, tol_t=1e-4, tol_r=1e-2, k=10
    )

    save_pose_json(R, t)
