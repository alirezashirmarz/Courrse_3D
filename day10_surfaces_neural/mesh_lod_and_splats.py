"""
Day 10 — Surfaces, mesh LoDs and Gaussian splats  (book Ch.10 & Ch.18)
---------------------------------------------------------------------
Run:  python mesh_lod_and_splats.py

Part A  Point cloud -> mesh (Poisson surface reconstruction, book Ch.10)
Part B  Mesh simplification -> LoD meshes (triangles vs bytes)
Part C  Why do Gaussian splats look "solid"?  A tiny splat renderer in NumPy:
        the SAME 3,000 points drawn as dots vs as soft 2D Gaussians.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import course3d as c3d

o3d = c3d.try_open3d()
if o3d is None:
    sys.exit("This session needs open3d:  pip install open3d")
OUT = c3d.out_dir(__file__)
c3d.banner("Day 10 · Meshes, LoD, Gaussian splats")

# ---------------- Part A: Poisson reconstruction ------------------------------------------------
# a "scanned" object: a sphere with bumps (like a rock)
rng = np.random.default_rng(5)
v = rng.normal(size=(30000, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
r = 1 + 0.15 * np.sin(5 * v[:, 0]) * np.cos(4 * v[:, 1]) + 0.08 * np.sin(9 * v[:, 2])
pts = v * r[:, None] + rng.normal(0, 0.005, (len(v), 3))
pcd = c3d.to_o3d(pts, 0.5 + 0.5 * v)
pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(20)
mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
mesh.compute_vertex_normals()
print(f"[A] Poisson: {len(pts):,} points -> mesh with {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
o3d.io.write_triangle_mesh(str(OUT / "rock_full.ply"), mesh)

# ---------------- Part B: LoD meshes --------------------------------------------------------------
print("\n[B] Mesh LoDs by quadric decimation")
print("    LoD  triangles   file size   (a streaming client picks one per object by distance)")
lods = []
for i, target in enumerate([len(mesh.triangles), 20000, 5000, 1000, 200]):
    m = mesh.simplify_quadric_decimation(target) if i else mesh
    m.compute_vertex_normals()
    f = OUT / f"rock_lod{i}.ply"; o3d.io.write_triangle_mesh(str(f), m, write_ascii=False)
    lods.append(m)
    print(f"    {i:3d}  {len(m.triangles):9,d}   {c3d.human_bytes(f.stat().st_size):>10s}")

fig = plt.figure(figsize=(18, 3.8))
for i, m in enumerate(lods):
    V, F = np.asarray(m.vertices), np.asarray(m.triangles)
    ax = fig.add_subplot(1, 5, i + 1, projection="3d")
    ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F, color="#c8a06a", edgecolor="#00000055" if len(F) < 6000 else "none", lw=0.2)
    ax.set_title(f"LoD {i}: {len(F):,} triangles"); ax.set_axis_off(); ax.set_box_aspect((1, 1, 1))
plt.tight_layout(); plt.savefig(OUT / "mesh_lods.png", dpi=100); plt.close()

# ---------------- Part C: Gaussian splats vs points --------------------------------------------------
def splat_render(points, colors, K, R, t, W, H, scale_world, mode="gauss"):
    """Draw each 3D point as a screen-space disk/Gaussian whose size shrinks with distance.
    Sorted back-to-front and alpha-blended (this is the core of 3DGS rendering)."""
    uv, z = c3d.project(points, K, R, t)
    ok = z > 0.05; uv, z, colors = uv[ok], z[ok], colors[ok]
    order = np.argsort(-z)                                   # back to front
    img = np.ones((H, W, 3))
    for (u, v), zz, c in zip(uv[order], z[order], colors[order]):
        sigma = max(0.7, K[0, 0] * scale_world / zz)         # pixels
        rad = int(3 * sigma) + 1
        u0, v0 = int(u), int(v)
        if u0 < -rad or v0 < -rad or u0 >= W + rad or v0 >= H + rad:
            continue
        ys, xs = np.mgrid[max(0, v0 - rad):min(H, v0 + rad + 1), max(0, u0 - rad):min(W, u0 + rad + 1)]
        if ys.size == 0:
            continue
        d2 = (xs - u) ** 2 + (ys - v) ** 2
        alpha = 0.9 * np.exp(-0.5 * d2 / sigma**2) if mode == "gauss" else (d2 <= 1.0) * 1.0
        img[ys, xs] = img[ys, xs] * (1 - alpha[..., None]) + c * alpha[..., None]
    return img

sub = rng.choice(len(pts), 3000, replace=False)
K = c3d.intrinsics(360, 360, 40); R, t = c3d.look_at((2.2, -3.2, 1.6), (0, 0, 0))
cols = (0.5 + 0.5 * v)[sub]
fig, ax = plt.subplots(1, 3, figsize=(14, 4.8))
ax[0].imshow(splat_render(pts[sub], cols, K, R, t, 360, 360, 0, mode="point")); ax[0].set_title("3,000 points as 1-pixel dots")
ax[1].imshow(splat_render(pts[sub], cols, K, R, t, 360, 360, 0.02)); ax[1].set_title("same points as small Gaussians")
ax[2].imshow(splat_render(pts[sub], cols, K, R, t, 360, 360, 0.045)); ax[2].set_title("same points as larger Gaussians\n(looks like a solid surface!)")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "gaussian_splats.png", dpi=100); plt.close()
print("\n[C] Real 3D Gaussian Splatting stores per splat: position(3) + scale(3) + rotation(4)"
      " + opacity(1) + colour as spherical harmonics(48) = 59 floats ≈ 236 bytes!  (vs 15 B for a coloured point)")
print("    -> compressing/streaming splats is a hot research topic in 6DoF streaming.")
print("\nsaved mesh_lods.png and gaussian_splats.png in", OUT)
c3d.show([mesh], "Day 10 — Poisson mesh (full)")
