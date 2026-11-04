"""
Day 09 — From geometry to meaning: features, RANSAC and clustering
------------------------------------------------------------------
Run:  python segment_scene.py

Geometry alone is "just points".  To stream SMARTLY we want to know
WHAT the points are (ground? house? person?) — e.g. send the person in
high quality, the ground in low quality.

Part A  Local shape features from PCA of neighbourhoods (book Ch.6-7):
        planarity / linearity / sphericity / verticality
Part B  RANSAC plane fitting from scratch (book Ch.9) -> find the ground
Part C  DBSCAN clustering (book Ch.12) -> split the rest into objects
Part D  Compare with the true labels (a tiny bit of evaluation)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import course3d as c3d

OUT = c3d.out_dir(__file__)
c3d.banner("Day 09 · Scene understanding")
rng = np.random.default_rng(0)
P, C, L = c3d.make_scene(80_000)
P = P + rng.normal(0, 0.01, P.shape)

# ---------------- Part A: PCA features -----------------------------------------------
tree = cKDTree(P)
_, nn = tree.query(P, k=50)
nb = P[nn] - P[nn].mean(1, keepdims=True)
cov = np.einsum("nki,nkj->nij", nb, nb) / 50
evals, evecs = np.linalg.eigh(cov)                  # ascending l3<=l2<=l1
l3, l2, l1 = evals[:, 0], evals[:, 1], evals[:, 2]
l1 = np.maximum(l1, 1e-12)
linearity = (l1 - l2) / l1
planarity = (l2 - l3) / l1
sphericity = l3 / l1
verticality = 1 - np.abs(evecs[:, 2, 0])            # normal = eigenvector of smallest eigenvalue
print("[A] mean features per TRUE class (high planarity = flat surface, high sphericity = scattered):")
print(f"    {'class':8s} {'linear':>7s} {'planar':>7s} {'spheric':>8s} {'vertical':>9s}")
for k in np.unique(L):
    m = L == k
    print(f"    {c3d.LABEL_NAMES[k]:8s} {linearity[m].mean():7.2f} {planarity[m].mean():7.2f} {sphericity[m].mean():8.2f} {verticality[m].mean():9.2f}")

# ---------------- Part B: RANSAC plane from scratch ----------------------------------------
def ransac_plane(pts, threshold=0.05, iterations=500, rng=rng):
    best_inliers, best_model = None, None
    for _ in range(iterations):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n); d = -n @ a
        dist = np.abs(pts @ n + d)
        inl = dist < threshold
        if best_inliers is None or inl.sum() > best_inliers.sum():
            best_inliers, best_model = inl, (n, d)
    return best_model, best_inliers

(n, d), ground = ransac_plane(P)
print(f"\n[B] RANSAC plane: normal {np.round(n, 3)}, d = {d:.3f}  -> {ground.sum():,} ground points")

# ---------------- Part C: DBSCAN on the rest -------------------------------------------------
rest = ~ground
o3d = c3d.try_open3d()
if o3d is not None:
    labels = np.array(c3d.to_o3d(P[rest]).cluster_dbscan(eps=0.5, min_points=10))
else:
    from sklearn.cluster import DBSCAN
    labels = DBSCAN(eps=0.5, min_samples=10).fit_predict(P[rest])
n_cl = labels.max() + 1
print(f"[C] DBSCAN found {n_cl} objects (+{(labels < 0).sum()} noise points)")
for k in range(n_cl):
    m = labels == k
    ext = np.ptp(P[rest][m], axis=0)
    true = np.bincount(L[rest][m]).argmax()
    print(f"    object {k}: {m.sum():6,d} pts, size {ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} m  (mostly '{c3d.LABEL_NAMES[true]}')")

# ---------------- Part D: evaluation -----------------------------------------------------------
gt_ground = L == 0
tp = (ground & gt_ground).sum(); prec = tp / ground.sum(); rec = tp / gt_ground.sum()
print(f"\n[D] Ground detection: precision {prec:.2%}, recall {rec:.2%}, IoU {tp / (ground | gt_ground).sum():.2%}")

# ---------------- figure + export -------------------------------------------------------------
pal = plt.cm.tab20(np.arange(20))[:, :3]
seg_col = np.tile([0.6, 0.6, 0.6], (len(P), 1))
cl_col = np.where(labels[:, None] >= 0, pal[labels % 20], [[0, 0, 0]])
seg_col[rest] = cl_col
eye, tgt = (18, -22, 12), (0, 2, 1)
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
for a, (col, ttl) in zip(ax, [(plt.cm.viridis(planarity)[:, :3], "Planarity (yellow = flat)"),
                              (plt.cm.magma(verticality)[:, :3], "Verticality (bright = vertical wall)"),
                              (seg_col, "RANSAC ground (grey) + DBSCAN objects")]):
    img, _ = c3d.render(P, col, c3d.intrinsics(600, 400, 60), *c3d.look_at(eye, tgt), 600, 400, 2)
    a.imshow(img); a.set_title(ttl); a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "segmentation.png", dpi=100); plt.close()
c3d.write_ply(OUT / "segmented.ply", P, seg_col)
print("\nsaved", OUT / "segmentation.png")
c3d.show([(P, seg_col)], "Day 09 — ground + objects")
