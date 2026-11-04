"""
Day 06 — How 3D is captured: the pinhole camera & RGB-D -> point cloud
---------------------------------------------------------------------
Run:  python pinhole_and_rgbd.py

Part A  Project a 3D point to a pixel by hand with K and [R|t].
Part B  Simulate an RGB-D sensor (like a Kinect / iPhone LiDAR): get a
        colour image + a depth image.
Part C  Back-project every pixel to 3D  ->  a point cloud.
Part D  Use the WRONG focal length and watch the 3D shape distort
        (book Ch.10: "without accurate intrinsics the point cloud will
        suffer from distortions").
Part E  Fuse two RGB-D views (different poses) into one cloud = the idea
        behind scanning / photogrammetry.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import course3d as c3d

np.set_printoptions(precision=2, suppress=True)
OUT = c3d.out_dir(__file__)
c3d.banner("Day 06 · Camera model & RGB-D")

# ---------- Part A: by hand ---------------------------------------------------------
W, H = 640, 480
K = c3d.intrinsics(W, H, fov_deg=60)
print("Intrinsic matrix K (fx, fy = focal length in pixels; cx, cy = image centre):\n", K)
eye, target = np.array([0.0, -12, 3]), np.array([0.0, 4, 1.5])
R, t = c3d.look_at(eye, target)
Xw = np.array([0.0, 4.0, 4.6])                       # roof ridge of the house
Xc = R @ Xw + t
u, v = K[0, 0] * Xc[0] / Xc[2] + K[0, 2], K[1, 1] * Xc[1] / Xc[2] + K[1, 2]
print(f"\nWorld point {Xw} -> camera coords {Xc} -> pixel (u, v) = ({u:.1f}, {v:.1f})")
print("Formula:  u = fx * X/Z + cx ,  v = fy * Y/Z + cy   (divide by depth Z = perspective!)")

# ---------- Part B: simulate an RGB-D sensor ----------------------------------------
P, C, _ = c3d.make_scene(400_000)
rgb, depth = c3d.render(P, C, K, R, t, W, H, point_px=3, bg=(0.8, 0.9, 1.0))
depth[depth > 30] = np.inf                      # sensor max range 30 m
print(f"\nRGB-D frame: colour {rgb.shape}, depth {depth.shape}, valid depth pixels {np.isfinite(depth).mean():.0%}")

# ---------- Part C: back-projection -------------------------------------------------
def backproject(depth, rgb, K, R, t, fx_override=None):
    fx = fx_override or K[0, 0]
    v, u = np.nonzero(np.isfinite(depth))
    z = depth[v, u]
    Xc = np.c_[(u - K[0, 2]) * z / fx, (v - K[1, 2]) * z / fx, z]
    return (Xc - t) @ R, rgb[v, u]           # camera -> world
pc, pc_col = backproject(depth, rgb, K, R, t)
print(f"Back-projected point cloud: {len(pc):,} points")
c3d.write_ply(OUT / "rgbd_cloud.ply", pc, pc_col)

# ---------- Part D: wrong intrinsics ------------------------------------------------
bad, bad_col = backproject(depth, rgb, K, R, t, fx_override=K[0, 0] * 0.6)
house = (np.abs(pc[:, 0]) < 3.3) & (pc[:, 1] > 1.8) & (pc[:, 1] < 6.3) & (pc[:, 2] > 0.3)   # same pixels in both clouds
print("House width with correct fx = %.0f px : %.2f m   (true: walls 6.0 m, roof overhang 6.4 m)" % (K[0, 0], np.ptp(pc[house, 0])))
print("House width with WRONG   fx = %.0f px : %.2f m   -> the 3D shape is distorted!" % (K[0, 0] * 0.6, np.ptp(bad[house, 0])))

# ---------- Part E: two views fused ----------------------------------------------------
R2, t2 = c3d.look_at([12, 14, 4], target)
rgb2, depth2 = c3d.render(P, C, K, R2, t2, W, H, point_px=3, bg=(0.8, 0.9, 1.0)); depth2[depth2 > 30] = np.inf
pc2, col2 = backproject(depth2, rgb2, K, R2, t2)
fused = np.vstack([pc, pc2]); fused_col = np.vstack([pc_col, col2])
c3d.write_ply(OUT / "fused_two_views.ply", fused, fused_col)
print(f"Fused 2 views -> {len(fused):,} points (back of the house now filled in)")

# ---------- figure ----------------------------------------------------------------
top = dict(eye=(0.01, 2, 45), target=(0, 2, 0))
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
ax[0, 0].imshow(rgb); ax[0, 0].set_title("Sensor colour image"); ax[0, 0].plot(u, v, "r+", ms=20, mew=3)
ax[0, 1].imshow(np.where(np.isfinite(depth), depth, np.nan), cmap="turbo"); ax[0, 1].set_title("Sensor depth image (m)")
for a, (pts, col, ttl) in zip([ax[0, 2], ax[1, 0], ax[1, 1], ax[1, 2]], [
        (pc, pc_col, "Back-projected cloud — side view"),
        (pc, pc_col, "Correct intrinsics — top view"),
        (bad, bad_col, "WRONG fx (x0.6) — top view: stretched!"),
        (fused, fused_col, "Two views fused — top view")]):
    kw = dict(eye=(14, -16, 9), target=(0, 3, 1)) if "side" in ttl else top
    img, _ = c3d.render(pts, col, c3d.intrinsics(480, 360, 60), *c3d.look_at(kw["eye"], kw["target"], up=(0, 1, 0) if kw is top else (0, 0, 1)), 480, 360, 2)
    a.imshow(img); a.set_title(ttl)
for a in ax.ravel(): a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "camera_model.png", dpi=100); plt.close()
print("\nsaved", OUT / "camera_model.png")
c3d.show([(fused, fused_col)], "Day 06 — fused RGB-D views")
