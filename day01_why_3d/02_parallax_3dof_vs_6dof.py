"""
Day 01 — 3DoF vs 6DoF: why moving your head matters
---------------------------------------------------
Run:  python 02_parallax_3dof_vs_6dof.py

A viewer has 6 degrees of freedom (DoF):
  3 for ROTATION   : yaw (look left/right), pitch (up/down), roll (tilt head)
  3 for TRANSLATION: x, y, z (move left/right, forward/back, up/down)

360° video gives you only the 3 rotations (3DoF).  If you lean to the side,
nothing changes - you cannot look *behind* the tree.  6DoF content
(point clouds, meshes, Gaussian splats...) lets you move: you see PARALLAX.

Output: outputs/parallax.png
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
c3d.banner("Day 01 · 3DoF vs 6DoF")
P, C, L = c3d.make_scene(150_000)
K = c3d.intrinsics(480, 320, 70)

# viewer stands at (0,-14) at eye height 1.7 m, looking north (+Y => yaw 90°)
base = dict(x=0.0, y=-14.0, z=1.7, yaw=np.radians(90), pitch=np.radians(-3), roll=0.0)
views = {
    "start pose":               {},
    "3DoF: turn head left 25°": dict(yaw=base["yaw"] + np.radians(25)),
    "3DoF: tilt head (roll 15°)": dict(roll=np.radians(15)),
    "6DoF: step 4 m to the right": dict(x=4.0),
    "6DoF: step 6 m forward":   dict(y=-8.0),
    "6DoF: crouch to 0.6 m":    dict(z=0.6),
}
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, (name, change) in zip(axes.ravel(), views.items()):
    pose = {**base, **change}
    R, t = c3d.pose_to_view(**pose)
    img, _ = c3d.render(P, C, K, R, t, 480, 320, point_px=2, bg=(0.85, 0.92, 1.0))
    ax.imshow(img); ax.axis("off")
    ax.set_title(f"{name}\n(x={pose['x']:.0f}, y={pose['y']:.0f}, z={pose['z']:.1f}, yaw={np.degrees(pose['yaw']):.0f}°, roll={np.degrees(pose['roll']):.0f}°)", fontsize=9)
    print(f"rendered: {name}")
plt.suptitle("Top row = rotation only (3DoF).  Bottom row = translation (the extra 3 of 6DoF) -> objects shift relative to each other (parallax)")
plt.tight_layout(); plt.savefig(OUT / "parallax.png", dpi=110); plt.close()
print("saved", OUT / "parallax.png")
print("\nLook at the bottom row: the tree/house/car change their RELATIVE positions. That is parallax,"
      " and it only exists in 6DoF.")
