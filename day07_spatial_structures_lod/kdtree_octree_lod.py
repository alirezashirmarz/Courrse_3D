"""
Day 07 — Organising space: k-d trees, octrees and Levels of Detail (LoD)
-----------------------------------------------------------------------
Run:  python kdtree_octree_lod.py

Why this matters for streaming: an OCTREE cuts space into 8 cubes again
and again.  Each level is a coarser/finer version of the scene  ->  that is
exactly a Level-of-Detail pyramid you can stream coarse-to-fine
(Potree, Cesium 3D Tiles and MPEG G-PCC all use this idea).

Part A  Nearest-neighbour search: brute force vs k-d tree (speed!)
Part B  Build an octree from scratch (book Ch.17 "condensed octree")
Part C  Turn octree levels into an LoD pyramid and measure bytes per level
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import course3d as c3d

OUT = c3d.out_dir(__file__)
c3d.banner("Day 07 · k-d tree, octree, LoD")
P, C, L = c3d.make_scene(300_000)
print(f"Scene: {len(P):,} points")

# ---------------- Part A: k-d tree ---------------------------------------------
queries = P[np.random.default_rng(1).choice(len(P), 200, replace=False)]
t0 = time.perf_counter()
bf = np.array([np.argmin(((P - q) ** 2).sum(1)) for q in queries])
t_bf = time.perf_counter() - t0
t0 = time.perf_counter(); tree = cKDTree(P); t_build = time.perf_counter() - t0
t0 = time.perf_counter(); _, kd = tree.query(queries, k=1); t_kd = time.perf_counter() - t0
print(f"\n[A] 200 nearest-neighbour queries")
print(f"    brute force : {t_bf*1000:8.1f} ms")
print(f"    k-d tree    : {t_kd*1000:8.2f} ms  (+ {t_build*1000:.0f} ms one-time build)  -> {t_bf/t_kd:,.0f}x faster, same answer: {np.all(bf == kd)}")

# ---------------- Part B: octree from scratch ----------------------------------------
class OctreeNode:
    __slots__ = ("center", "size", "depth", "idx", "children")
    def __init__(self, center, size, depth, idx):
        self.center, self.size, self.depth, self.idx, self.children = center, size, depth, idx, []

def build_octree(points, max_depth=8, min_points=8):
    lo, hi = points.min(0), points.max(0)
    root = OctreeNode((lo + hi) / 2, (hi - lo).max() * 1.0001, 0, np.arange(len(points)))
    stack = [root]
    while stack:
        n = stack.pop()
        if n.depth >= max_depth or len(n.idx) <= min_points:
            continue
        p = points[n.idx]
        octant = ((p[:, 0] > n.center[0]).astype(int) << 2) | ((p[:, 1] > n.center[1]).astype(int) << 1) | (p[:, 2] > n.center[2]).astype(int)
        for o in range(8):
            sel = n.idx[octant == o]
            if len(sel) == 0:
                continue                         # empty space is never stored!
            off = (np.array([(o >> 2) & 1, (o >> 1) & 1, o & 1]) - 0.5) * n.size / 2
            child = OctreeNode(n.center + off, n.size / 2, n.depth + 1, sel)
            n.children.append(child); stack.append(child)
    return root

t0 = time.perf_counter(); root = build_octree(P, max_depth=10); t_oct = time.perf_counter() - t0
per_level = {}
stack = [root]
while stack:
    n = stack.pop(); per_level[n.depth] = per_level.get(n.depth, 0) + 1; stack.extend(n.children)
print(f"\n[B] Octree built in {t_oct:.2f} s. Root cube size = {root.size:.1f} m")
print("    level  cell size   occupied cells   (of 8^level possible)")
for d in sorted(per_level):
    print(f"    {d:5d}  {root.size / 2**d:8.3f} m  {per_level[d]:12,d}     {per_level[d] / 8**d:8.4%}")
print("    -> most of space is EMPTY; the octree only stores occupied cells.")

# ---------------- Part C: LoD pyramid from octree levels -------------------------------
def lod_from_level(points, colors, level, root):
    """One representative point per occupied cell at this level (cell centroid)."""
    cell = root.size / 2**level
    lo = root.center - root.size / 2
    keys = np.floor((points - lo) / cell).astype(np.int64)
    k1 = (keys[:, 0] << 42) | (keys[:, 1] << 21) | keys[:, 2]
    uniq, inv = np.unique(k1, return_inverse=True)
    cnt = np.bincount(inv)
    pts = np.c_[[np.bincount(inv, points[:, i]) / cnt for i in range(3)]].T
    col = np.c_[[np.bincount(inv, colors[:, i]) / cnt for i in range(3)]].T
    return pts, col

fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
print("\n[C] LoD pyramid (what a streaming client would download, coarse -> fine)")
print("    LoD  level   points      bytes(15 B/pt)   cumulative")
cum = 0
for i, lvl in enumerate([4, 5, 6, 7, 9]):
    pts, col = lod_from_level(P, C, lvl, root)
    b = len(pts) * 15; cum += b
    print(f"    {i:3d}  {lvl:5d}  {len(pts):9,d}   {c3d.human_bytes(b):>12s}   {c3d.human_bytes(cum):>12s}")
    c3d.write_ply(OUT / f"lod{i}.ply", pts, col)
    K = c3d.intrinsics(400, 300, 60); R, t = c3d.look_at((18, -22, 10), (0, 2, 1))
    img, _ = c3d.render(pts, col, K, R, t, 400, 300, point_px=max(2, 9 - 2 * i))
    axes[i].imshow(img); axes[i].axis("off"); axes[i].set_title(f"LoD {i} (octree level {lvl})\n{len(pts):,} pts · {c3d.human_bytes(b)}")
plt.tight_layout(); plt.savefig(OUT / "lod_pyramid.png", dpi=100); plt.close()
print("\nsaved", OUT / "lod_pyramid.png", "and lod0..lod4.ply")
print("Streaming idea: send LoD0 first (tiny, instant preview), then refine where the viewer is looking.")

o3d = c3d.try_open3d()
if o3d is not None and c3d.viz_enabled():
    oc = o3d.geometry.Octree(max_depth=6); oc.convert_from_point_cloud(c3d.to_o3d(P, C), size_expand=0.01)
    o3d.visualization.draw_geometries([oc], window_name="Day 07 — Open3D octree (depth 6)")
