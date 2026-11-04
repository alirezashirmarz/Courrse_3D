"""
Day 01 — What is "3D" and why do we need it?
--------------------------------------------
Run:  python 01_dimensions.py

We describe the SAME object (a hill with a ball on top) four ways:
  1D  a single profile line           -> one number per x
  2D  a photo (colour image)           -> colour per (x, y) pixel, no depth
  2.5D a depth map                     -> ONE depth per pixel (cannot see behind)
  3D  a point cloud                    -> free (x, y, z) points, any side

Output: outputs/dimensions.png  (open it!)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import course3d as c3d

OUT = c3d.out_dir(__file__)
c3d.banner("Day 01 · Dimensions: 1D -> 2D -> 2.5D -> 3D")

# --- build a 3D object: a hill (surface) + a ball floating above it --------
g = np.linspace(-3, 3, 160)
X, Y = np.meshgrid(g, g)
Z = 1.5 * np.exp(-(X**2 + Y**2) / 3)
hill = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
hill_col = np.c_[0.3 + 0.2 * Z.ravel() / 1.5, 0.6 * np.ones(hill.shape[0]), 0.3 * np.ones(hill.shape[0])]
ball, ball_col = c3d.sample_sphere((0, 0, 2.6), 0.5, 6000, [0.9, 0.3, 0.1])
P = np.vstack([hill, ball]); C = np.vstack([hill_col, ball_col])
print(f"3D object = {len(P):,} points, each with (x, y, z) + (r, g, b)")

# --- 1D: a single profile through the middle --------------------------------
profile = Z[80, :]

# --- 2D + 2.5D: a camera looking down from above ----------------------------
K = c3d.intrinsics(320, 320, 50)
R, t = c3d.look_at((0.001, 0, 9), (0, 0, 0), up=(0, 1, 0))
img, depth = c3d.render(P, C, K, R, t, 320, 320, point_px=3)
print("2D image shape:", img.shape, "-> colour only, depth is lost")
hidden = np.isfinite(depth).sum()
print(f"2.5D depth map keeps ONE depth for {hidden:,} pixels -> the hill under the ball is invisible")

# --- figure -----------------------------------------------------------------
fig = plt.figure(figsize=(16, 4.4))
ax = fig.add_subplot(1, 4, 1); ax.plot(g, profile, lw=3); ax.set_title("1D: height along one line"); ax.set_xlabel("x"); ax.set_ylabel("z")
ax = fig.add_subplot(1, 4, 2); ax.imshow(img); ax.set_title("2D: a photo (colour per pixel)"); ax.axis("off")
ax = fig.add_subplot(1, 4, 3); d = np.where(np.isfinite(depth), depth, np.nan); im = ax.imshow(d, cmap="viridis_r"); ax.set_title("2.5D: depth map (1 depth / pixel)"); ax.axis("off"); fig.colorbar(im, ax=ax, fraction=0.046, label="distance to camera (m)")
ax = fig.add_subplot(1, 4, 4, projection="3d"); s = np.random.default_rng(0).choice(len(P), 6000, replace=False)
ax.scatter(P[s, 0], P[s, 1], P[s, 2], c=C[s], s=2); ax.set_title("3D: point cloud (x, y, z)"); ax.view_init(18, -60)
plt.tight_layout(); plt.savefig(OUT / "dimensions.png", dpi=110); plt.close()
print("saved", OUT / "dimensions.png")

# --- an interactive 3D window (rotate with the mouse!) ----------------------
c3d.show([(P, C)], "Day 01 — the same object in true 3D", OUT / "object3d.png")
print("\nQuestion to think about: from the top-down depth map, can you recover the hill UNDER the ball? Why not?")
