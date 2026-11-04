"""
Day 05 — One object, many representations  (book Ch.4)
------------------------------------------------------
Run:  python representations.py

point cloud  <->  mesh  <->  voxels  <->  depth map (2.5D raster)
We convert between them (the book's "canonical link") and compare
memory, detail and what each one is good for.
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
c3d.banner("Day 05 · 3D representations")

# --- 1) MESH: vertices + triangles ------------------------------------------------
mesh = o3d.geometry.TriangleMesh.create_torus(torus_radius=1.0, tube_radius=0.4, radial_resolution=60, tubular_resolution=30)
mesh.compute_vertex_normals(); mesh.paint_uniform_color([0.95, 0.55, 0.15])
V, F = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
print(f"MESH        : {len(V):6,} vertices, {len(F):6,} triangles  -> ~{V.nbytes + F.nbytes:,} bytes")
print("              surface is CONTINUOUS: every triangle connects 3 vertices (topology!)")

# --- 2) mesh -> POINT CLOUD (sampling) ---------------------------------------------
pcd = mesh.sample_points_poisson_disk(8000)
pcd.paint_uniform_color([0.2, 0.4, 0.9])
pts = np.asarray(pcd.points)
print(f"POINT CLOUD : {len(pts):6,} points, no connectivity      -> ~{len(pts)*12:,} bytes")

# --- 3) point cloud -> VOXELS ---------------------------------------------------------
for vs in [0.3, 0.15, 0.05]:
    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=vs)
    print(f"VOXELS      : voxel size {vs:4.2f} m -> {len(vg.get_voxels()):6,} occupied voxels (like 3D Lego)")
vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.15)
# voxels -> point cloud: take voxel centres
centres = np.array([vg.get_voxel_center_coordinate(v.grid_index) for v in vg.get_voxels()])
print(f"VOXELS -> PC: {len(centres):,} voxel centres (information lost below 0.15 m!)")

# --- 4) point cloud -> DEPTH MAP (2.5D) and back --------------------------------------
W, H = 200, 200
K = c3d.intrinsics(W, H, 55)
R, t = c3d.look_at((0, -0.001, 5), (0, 0, 0), up=(0, 1, 0))
dense = np.asarray(mesh.sample_points_uniformly(80000).points)
_, depth = c3d.render(dense, np.zeros_like(dense), K, R, t, W, H, point_px=2)
vv, uu = np.nonzero(np.isfinite(depth))
z = depth[vv, uu]
Xc = np.c_[(uu - K[0, 2]) * z / K[0, 0], (vv - K[1, 2]) * z / K[1, 1], z]   # back-project
back = (Xc - t) @ R                                                       # camera -> world
print(f"DEPTH MAP   : {W}x{H} pixels, {len(z):,} valid -> back to {len(back):,} points (only the visible TOP half!)")

# --- figure --------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 4.2))
ax = fig.add_subplot(1, 4, 1, projection="3d"); ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F, color="#f08a24", edgecolor="#00000022", lw=0.1); ax.set_title("Mesh (triangles)")
ax = fig.add_subplot(1, 4, 2, projection="3d"); ax.scatter(*pts.T, s=1, c="#2f6fed"); ax.set_title("Point cloud (samples)")
ax = fig.add_subplot(1, 4, 3, projection="3d"); ax.scatter(*centres.T, s=12, marker="s", c="#6a3fbf"); ax.set_title("Voxels 0.15 m (centres)")
ax = fig.add_subplot(1, 4, 4); im = ax.imshow(np.where(np.isfinite(depth), depth, np.nan), cmap="viridis_r"); ax.set_title("Depth map from above (2.5D)"); ax.axis("off")
for a in fig.axes[:3]:
    a.set_box_aspect((1, 1, 0.45)); a.set_axis_off()
plt.tight_layout(); plt.savefig(OUT / "representations.png", dpi=110); plt.close()
o3d.io.write_triangle_mesh(str(OUT / "torus_mesh.ply"), mesh)
o3d.io.write_point_cloud(str(OUT / "torus_points.ply"), pcd)
print("\nsaved", OUT / "representations.png")

if c3d.viz_enabled():
    vg.translate([3.5, 0, 0]); pcd.translate([-3.5, 0, 0])
    c3d.show([mesh, pcd, vg], "Day 05 — point cloud | mesh | voxels")
