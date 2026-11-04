"""
Day 11 — How a 6DoF viewer decides what to draw: the view frustum
-----------------------------------------------------------------
Run:  python frustum_culling.py

A camera only sees a pyramid-shaped volume of space: the VIEW FRUSTUM
(near plane, far plane, 4 side planes from the field of view).
Everything outside can be skipped — not rendered, and (Day 14!) not even
downloaded.  We test this for a few 6DoF poses and measure how much of
the scene is actually visible.

Also writes viewer6dof/scene.ply for the browser viewer in this folder.
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
c3d.banner("Day 11 · View frustum & 6DoF viewing")
P, C, L = c3d.make_scene(250_000)

# write the scene for the browser viewer (three.js uses Y-UP: (x, y, z) -> (x, z, -y))
web = Path(__file__).resolve().parent / "viewer6dof"
c3d.write_ply(web / "scene.ply", np.c_[P[:, 0], P[:, 2], -P[:, 1]], C)
print("wrote", web / "scene.ply", "(converted Z-up -> Y-up for the browser)")

def frustum_mask(points, pose, fov_deg=70, aspect=16 / 9, near=0.1, far=40):
    """True for points inside the camera frustum of a 6DoF pose."""
    R, t = c3d.pose_to_view(**pose)
    Xc = points @ R.T + t                          # into camera coordinates
    z = Xc[:, 2]
    tx = np.tan(np.radians(fov_deg) / 2); ty = tx / aspect
    return (z > near) & (z < far) & (np.abs(Xc[:, 0]) < z * tx) & (np.abs(Xc[:, 1]) < z * ty)

poses = {
    "A: street corner, looking at house": dict(x=-14, y=-14, z=1.7, yaw=np.radians(45), pitch=0, roll=0),
    "B: close to the door":               dict(x=0, y=-4, z=1.7, yaw=np.radians(90), pitch=0, roll=0),
    "C: looking away (south)":            dict(x=0, y=-4, z=1.7, yaw=np.radians(-90), pitch=0, roll=0),
    "D: drone, looking down":             dict(x=0, y=-10, z=25, yaw=np.radians(90), pitch=np.radians(-60), roll=0),
}
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for i, (name, pose) in enumerate(poses.items()):
    m = frustum_mask(P, pose)
    print(f"{name:38s}: visible {m.sum():7,d} / {len(P):,} points = {m.mean():5.1%}")
    ax = axes[0, i]
    ax.scatter(P[~m, 0], P[~m, 1], s=0.2, c="#cccccc"); ax.scatter(P[m, 0], P[m, 1], s=0.2, c=C[m])
    ax.plot(pose["x"], pose["y"], "r^", ms=12)
    ax.arrow(pose["x"], pose["y"], 4 * np.cos(pose["yaw"]), 4 * np.sin(pose["yaw"]), color="r", width=0.3)
    ax.set_aspect("equal"); ax.set_title(f"{name}\nvisible = {m.mean():.0%} of points (top view)", fontsize=10)
    R, t = c3d.pose_to_view(**pose)
    img, _ = c3d.render(P[m], C[m], c3d.intrinsics(480, 270, 70), R, t, 480, 270, 2, bg=(0.85, 0.92, 1))
    axes[1, i].imshow(img); axes[1, i].axis("off"); axes[1, i].set_title("what the viewer sees")
plt.tight_layout(); plt.savefig(OUT / "frustum.png", dpi=90); plt.close()
print("\nsaved", OUT / "frustum.png")
print("Big idea: at any moment a 6DoF viewer sees only a FRACTION of the scene ->"
      " send that fraction first (viewport-adaptive streaming, Day 14).")
print("\nNow try the browser viewer:\n   cd viewer6dof\n   python -m http.server 8000\n   open http://localhost:8000")
