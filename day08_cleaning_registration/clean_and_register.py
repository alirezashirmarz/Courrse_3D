"""
Day 08 — Cleaning point clouds and aligning them (registration = finding a 6DoF pose!)
-------------------------------------------------------------------------------------
Run:  python clean_and_register.py

Part A  Real sensors are noisy: add noise + outliers, then clean them
        (voxel downsampling + statistical outlier removal, book Ch.5-6)
Part B  Normals: which way does each surface point face?
Part C  Registration with ICP (Iterative Closest Point, book Ch.6).
        We move a scan by a KNOWN 6DoF pose and ask ICP (point-to-plane)
        to find it again. Exercise: switch to PointToPoint and compare!
        Same maths is used by AR headsets / robots to track their own 6DoF pose.
"""
import sys, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import course3d as c3d

o3d = c3d.try_open3d()
if o3d is None:
    sys.exit("This session needs open3d:  pip install open3d")
np.set_printoptions(precision=3, suppress=True)
OUT = c3d.out_dir(__file__)
c3d.banner("Day 08 · Cleaning & registration")
rng = np.random.default_rng(3)

# ---------------- Part A: noise, outliers, cleaning ------------------------------------
P, C, _ = c3d.make_house(40_000)
noisy = P + rng.normal(0, 0.02, P.shape)                       # 2 cm sensor noise
outliers = rng.uniform(P.min(0) - 2, P.max(0) + 2, (2000, 3))  # "flying pixels"
raw = c3d.to_o3d(np.vstack([noisy, outliers]), np.vstack([C, np.tile([1, 0, 1], (2000, 1))]))
print(f"[A] raw scan: {len(raw.points):,} points (incl. 2,000 magenta outliers)")

down = raw.voxel_down_sample(voxel_size=0.05)
print(f"    voxel downsample 5 cm -> {len(down.points):,} points")
clean, keep = down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print(f"    statistical outlier removal -> {len(clean.points):,} points ({len(down.points) - len(clean.points):,} removed)")
o3d.io.write_point_cloud(str(OUT / "house_raw.ply"), raw)
o3d.io.write_point_cloud(str(OUT / "house_clean.ply"), clean)

# ---------------- Part B: normals ------------------------------------------------------
clean.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
clean.orient_normals_towards_camera_location(np.array([0, -20, 5]))
N = np.asarray(clean.normals)
print(f"[B] normals estimated. {np.mean(np.abs(N[:, 2]) > 0.9):.0%} point straight up/down, "
      f"{np.mean(np.abs(N[:, 2]) < 0.1):.0%} are vertical walls")

# ---------------- Part C: ICP registration ---------------------------------------------
source = copy.deepcopy(clean)
true_T = np.eye(4)
true_T[:3, :3] = c3d.euler_to_R(np.radians(12), np.radians(3), np.radians(-2))
true_T[:3, 3] = [0.8, -0.5, 0.2]
target = copy.deepcopy(clean).transform(true_T)
target.points = o3d.utility.Vector3dVector(np.asarray(target.points) + rng.normal(0, 0.01, (len(target.points), 3)))
print(f"\n[C] We moved the scan by a secret 6DoF pose:\n{true_T}")

target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))   # point-to-plane ICP needs target normals
source.paint_uniform_color([0.95, 0.55, 0.15]); target.paint_uniform_color([0.18, 0.43, 0.93])
before = o3d.pipelines.registration.evaluate_registration(source, target, 0.3)
print(f"    before ICP: overlap (fitness) {before.fitness:.2f}, RMSE {before.inlier_rmse:.3f} m")
history = []
T = np.eye(4)
for it in range(60):          # run ICP one step at a time so we can watch it converge
    res = o3d.pipelines.registration.registration_icp(
        source, target, 0.5, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(),  # try ...PointToPoint() too!
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
    T = res.transformation
    history.append(res.inlier_rmse)
print(f"    after  ICP: overlap (fitness) {res.fitness:.2f}, RMSE {res.inlier_rmse:.3f} m")
print(f"    ICP found pose:\n{T}")
err_t = np.linalg.norm(T[:3, 3] - true_T[:3, 3])
err_R = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3].T @ true_T[:3, :3]) - 1) / 2, -1, 1)))
print(f"    error vs truth: translation {err_t*100:.1f} cm, rotation {err_R:.2f}°")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.figure(figsize=(6, 3.5)); plt.plot(history, "o-"); plt.xlabel("ICP iteration"); plt.ylabel("inlier RMSE (m)"); plt.title("ICP converging to the 6DoF pose"); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT / "icp_convergence.png", dpi=110); plt.close()
aligned = copy.deepcopy(source).transform(T)
both = np.vstack([np.asarray(aligned.points), np.asarray(target.points)])
cols = np.vstack([np.asarray(aligned.colors), np.asarray(target.colors)])
c3d.preview_png(np.vstack([np.asarray(source.points), np.asarray(target.points)]), cols, OUT / "before_icp.png", title="before ICP")
c3d.preview_png(both, cols, OUT / "after_icp.png", title="after ICP")
print("\nsaved before_icp.png / after_icp.png / icp_convergence.png in", OUT)
c3d.show([raw], "Day 08 — RAW (noise + magenta outliers)")
c3d.show([clean], "Day 08 — CLEANED")
c3d.show([source, target], "Day 08 — before ICP (orange = source, blue = target)")
c3d.show([aligned, target], "Day 08 — after ICP")
