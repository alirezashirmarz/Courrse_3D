"""
Day 16 — Predicting where the viewer will look (6DoF pose prediction)
--------------------------------------------------------------------
Run:  python pose_prediction.py                       (synthetic head motion)
      python pose_prediction.py my_pose_trace.csv     (trace recorded in the Day 11 viewer!)

A 6DoF streaming client must request content BEFORE the viewer needs it
(the network takes time).  So it must predict the future pose:
     "where will I be and where will I look in 300 ms?"
We compare three predictors at different horizons (= network latency):
  * no prediction      (use the current pose)
  * constant velocity  (keep moving the way you are moving)
  * linear regression  (fit a line to the last 0.3 s)
and show how prediction error grows with latency.
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
c3d.banner("Day 16 · Pose prediction")
DT = 1 / 60

if len(sys.argv) > 1:                                   # recorded trace from the browser viewer
    raw = np.genfromtxt(sys.argv[1], delimiter=",", names=True)
    t = raw["t_ms"] / 1000
    grid = np.arange(t[0], t[-1], DT)
    pos = np.c_[[np.interp(grid, t, raw[k]) for k in ("x", "y", "z")]].T
    yaw = np.interp(grid, t, np.unwrap(np.radians(raw["yaw"])))
    print(f"loaded {len(raw)} poses from {sys.argv[1]} ({t[-1]-t[0]:.1f} s)")
else:                                                   # synthetic: someone walking & looking around
    grid = np.arange(0, 30, DT)
    rng = np.random.default_rng(2)
    pos = np.c_[1.2 * grid, 2 * np.sin(grid / 3), 1.7 + 0.03 * np.sin(2 * np.pi * 1.8 * grid)]   # walk + head bob
    yaw = 0.8 * np.sin(grid / 2) + 0.4 * np.sin(grid * 1.3) + np.cumsum(rng.normal(0, 0.004, len(grid)))
    print("using a synthetic 30 s head-motion trace (walk + look around)")

def predict(i, h, method, win=18):
    if method == "no prediction": return pos[i], yaw[i]
    if method == "constant velocity":
        v = (pos[i] - pos[i - 1]) / DT; w = (yaw[i] - yaw[i - 1]) / DT
        return pos[i] + v * h, yaw[i] + w * h
    if method == "linear regression":
        tt = np.arange(-win + 1, 1) * DT
        A = np.c_[tt, np.ones_like(tt)]
        cp = np.linalg.lstsq(A, pos[i - win + 1:i + 1], rcond=None)[0]
        cy = np.linalg.lstsq(A, yaw[i - win + 1:i + 1], rcond=None)[0]
        return cp[0] * h + cp[1], cy[0] * h + cy[1]

horizons_ms = [50, 100, 200, 300, 500, 800]
methods = ["no prediction", "constant velocity", "linear regression"]
res = {m: dict(pos=[], ang=[]) for m in methods}
for hms in horizons_ms:
    k = int(round(hms / 1000 / DT))
    idx = np.arange(30, len(grid) - k)
    for m in methods:
        pe, ae = [], []
        for i in idx[::3]:
            p, y = predict(i, k * DT, m)
            pe.append(np.linalg.norm(p - pos[i + k])); ae.append(np.degrees(abs(y - yaw[i + k])))
        res[m]["pos"].append(np.mean(pe)); res[m]["ang"].append(np.mean(ae))
print(f"\n{'horizon':>8s} | " + " | ".join(f"{m:>22s}" for m in methods))
for j, h in enumerate(horizons_ms):
    print(f"{h:6d}ms | " + " | ".join(f"{res[m]['pos'][j]*100:6.1f} cm {res[m]['ang'][j]:6.1f}°   " for m in methods))

fig, ax = plt.subplots(1, 3, figsize=(17, 4.3))
ax[0].plot(grid, np.degrees(yaw)); ax[0].set_title("viewer yaw over time"); ax[0].set_xlabel("s"); ax[0].set_ylabel("yaw (°)")
for m in methods:
    ax[1].plot(horizons_ms, np.array(res[m]["pos"]) * 100, "o-", label=m)
    ax[2].plot(horizons_ms, res[m]["ang"], "o-", label=m)
ax[1].set_title("position error vs prediction horizon"); ax[1].set_ylabel("cm")
ax[2].set_title("viewing-direction error vs horizon"); ax[2].set_ylabel("degrees")
for a in ax[1:]: a.set_xlabel("horizon = latency (ms)"); a.grid(alpha=.3); a.legend()
plt.tight_layout(); plt.savefig(OUT / "pose_prediction.png", dpi=100); plt.close()
print("\nsaved", OUT / "pose_prediction.png")
print("Takeaway: prediction helps a lot for short horizons; for long latency, fetch a WIDER margin around the view.")
