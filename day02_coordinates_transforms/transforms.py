"""
Day 02 — Coordinates, transformations and the 6DoF pose
-------------------------------------------------------
Run:  python transforms.py

Key ideas you will SEE in the printout and in outputs/transforms.png:
  1. A point is just 3 numbers; a point cloud is an (N, 3) array.
  2. Translation = add a vector.  Rotation = multiply by a 3x3 matrix.
  3. A 4x4 "homogeneous" matrix packs rotation + translation together.
  4. ORDER MATTERS: rotate-then-move != move-then-rotate.
  5. A 6DoF pose = 3 translation numbers + 3 rotation numbers -> one 4x4 matrix.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import course3d as c3d

np.set_printoptions(precision=3, suppress=True)
OUT = c3d.out_dir(__file__)
c3d.banner("Day 02 · Coordinates & transformations")

# 1) points --------------------------------------------------------------------
p = np.array([1.0, 2.0, 3.0])
print("A point p =", p, "-> x=1, y=2, z=3 (metres)")
house, hcol, _ = c3d.make_house(4000)
print("A point cloud is an array of shape", house.shape)

# 2) translation and rotation ---------------------------------------------------
t = np.array([10.0, 0.0, 0.0])
print("\nTranslate p by", t, "->", p + t)

def Rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

R = Rz(90)
print("Rotation about Z by 90°:\n", R)
print("R @ [1,0,0] =", R @ [1, 0, 0], " (x-axis turned into y-axis)")
print("Is R a valid rotation?  R^T R = I ->", np.allclose(R.T @ R, np.eye(3)), "; det(R) =", round(np.linalg.det(R), 3))

# 3) homogeneous 4x4 -----------------------------------------------------------
def make_T(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T

def apply_T(T, pts):
    ph = np.c_[pts, np.ones(len(pts))]       # (N,4): add the "1"
    return (ph @ T.T)[:, :3]

T = make_T(R, t)
print("\nHomogeneous matrix T = [R | t]:\n", T)

# 4) order matters --------------------------------------------------------------
T_rot, T_move = make_T(Rz(90), [0, 0, 0]), make_T(np.eye(3), [10, 0, 0])
A = apply_T(T_move @ T_rot, house)   # rotate first, then move
B = apply_T(T_rot @ T_move, house)   # move first, then rotate
print("\nCentre after rotate->move:", A.mean(0), "   after move->rotate:", B.mean(0), " (different!)")

# 5) a 6DoF pose ----------------------------------------------------------------
def pose_matrix(x, y, z, yaw_deg, pitch_deg, roll_deg):
    return make_T(c3d.euler_to_R(*np.radians([yaw_deg, pitch_deg, roll_deg])), [x, y, z])

pose = pose_matrix(5, 5, 0, 45, 0, 0)
print("\n6DoF pose (x=5,y=5,z=0, yaw=45°) as a 4x4 matrix:\n", pose)
print("Inverse pose (world -> object):\n", np.linalg.inv(pose))
back = apply_T(np.linalg.inv(pose), apply_T(pose, house))
print("Apply pose then its inverse -> max error:", np.abs(back - house).max())

# quaternion: the compact way engines/streaming protocols send rotations
def R_to_quat(R):
    w = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = (R[2, 1] - R[1, 2]) / (4 * w); y = (R[0, 2] - R[2, 0]) / (4 * w); z = (R[1, 0] - R[0, 1]) / (4 * w)
    return np.array([w, x, y, z])
q = R_to_quat(pose[:3, :3])
print("Same rotation as a quaternion (w,x,y,z):", q, " |q| =", round(np.linalg.norm(q), 3))
print("=> A 6DoF pose can be sent over the network as 7 floats: position(3) + quaternion(4) = 28 bytes")

# figure -------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 5))
def scat(ax, pts, col, label):
    s = slice(None, None, 4)
    ax.scatter(pts[s, 0], pts[s, 1], pts[s, 2], c=col[s], s=1, label=label)
for i, (title, sets) in enumerate([
        ("Translation (+10 m in x)", [(house, hcol, "original"), (house + t, np.tile([0.2, 0.4, 0.9], (len(house), 1)), "moved")]),
        ("Order matters", [(house, hcol, "original"), (A, np.tile([0.1, 0.6, 0.2], (len(house), 1)), "rotate→move"), (B, np.tile([0.9, 0.3, 0.1], (len(house), 1)), "move→rotate")]),
        ("Several 6DoF poses of one object", [(apply_T(pose_matrix(8 * np.cos(a), 8 * np.sin(a), 0, np.degrees(a) + 90, 0, 0), house), np.tile(plt.cm.viridis(k / 6)[:3], (len(house), 1)), None) for k, a in enumerate(np.linspace(0, 2 * np.pi, 6, endpoint=False))])]):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    for pts, col, lab in sets:
        scat(ax, pts, col, lab)
    ax.quiver(0, 0, 0, 3, 0, 0, color="r"); ax.quiver(0, 0, 0, 0, 3, 0, color="g"); ax.quiver(0, 0, 0, 0, 0, 3, color="b")
    ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 0.4))
    if i < 2: ax.legend(loc="upper left", fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "transforms.png", dpi=110); plt.close()
print("\nsaved", OUT / "transforms.png", "(red/green/blue arrows = x/y/z axes)")
