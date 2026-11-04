"""
Day 14 — Viewport-adaptive 6DoF streaming: tiles x LoDs x pose
--------------------------------------------------------------
Run:  python viewport_adaptive.py

The core recipe of 6DoF streaming (Potree, Cesium 3D Tiles, MPEG-I):
  1. cut the scene into TILES (here a 2D grid of 5 m cells),
  2. store every tile at several LEVELS OF DETAIL (Day 7),
  3. every frame, look at the viewer's 6DoF POSE:
        - tile outside the frustum      -> don't send it
        - tile far away                 -> coarse LoD is enough
        - tile close and in view        -> fine LoD
  4. respect the BANDWIDTH budget: spend bytes where they help most.

We fly a camera along a path and compare 4 strategies:
  S1 "send everything"     full scene at max LoD
  S2 "frustum only"        visible tiles at max LoD
  S3 "frustum + distance"  visible tiles, LoD chosen by distance
  S4 "budgeted greedy"     S3 but never more than B bytes per update
and measure BYTES and IMAGE QUALITY (PSNR vs. the full-quality render).
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
c3d.banner("Day 14 · Viewport-adaptive streaming")
P, C, _ = c3d.make_scene(400_000, extent=20)
BYTES_PER_POINT = 15

# ---------------- 1+2: tiles and LoDs ---------------------------------------------------
TILE = 5.0
LOD_CELLS = [1.2, 0.6, 0.3, 0.12, 0.0]          # voxel size per LoD (0 = all points)
POINT_PX = [9, 7, 5, 3, 2]                     # on-screen point size per LoD (fills the gaps)
tile_ix = np.floor((P[:, :2] + 20) / TILE).astype(int)
tiles = {}
for key in set(map(tuple, tile_ix)):
    m = np.all(tile_ix == key, axis=1)
    pts, col = P[m], C[m]
    lods = []
    for cell in LOD_CELLS:
        if cell == 0:
            lods.append((pts, col)); continue
        k = np.floor(pts / cell).astype(np.int64)
        _, first = np.unique(k, axis=0, return_index=True)       # one point per voxel
        lods.append((pts[first], col[first]))
    lo, hi = pts.min(0), pts.max(0)
    tiles[key] = dict(lods=lods, lo=lo, hi=hi, center=(lo + hi) / 2, radius=np.linalg.norm(hi - lo) / 2)
print(f"{len(tiles)} tiles x {len(LOD_CELLS)} LoDs.  Points per LoD (whole scene): "
      + ", ".join(f"LoD{i}={sum(len(t['lods'][i][0]) for t in tiles.values()):,}" for i in range(len(LOD_CELLS))))

# ---------------- camera path ------------------------------------------------------------
W, H, FOV = 400, 240, 70
K = c3d.intrinsics(W, H, FOV)
ts = np.linspace(0, 1, 40)
path = [dict(x=-16 + 28 * s, y=-14 + 10 * s, z=1.7 + 2 * np.sin(np.pi * s), yaw=np.radians(70 - 60 * s), pitch=np.radians(-5), roll=0) for s in ts]

def tile_visible(tile, R, t):
    """Sphere-vs-frustum test (conservative)."""
    c = R @ tile["center"] + t
    r = tile["radius"]
    tx = np.tan(np.radians(FOV) / 2); ty = tx * H / W
    if c[2] < -r: return False
    for nx, nz in [(1, tx), (-1, tx)]:                  # left/right planes
        if (nx * c[0] - nz * c[2]) / np.hypot(1, nz) > r: return False
    for ny, nz in [(1, ty), (-1, ty)]:                  # top/bottom planes
        if (ny * c[1] - nz * c[2]) / np.hypot(1, nz) > r: return False
    return True

def lod_for_distance(d):
    return 4 if d < 12 else 3 if d < 20 else 2 if d < 28 else 1 if d < 38 else 0

def choose(strategy, pose, budget=None):
    R, t = c3d.pose_to_view(**pose)
    eye = np.array([pose["x"], pose["y"], pose["z"]])
    sel = {}
    for k, tl in tiles.items():
        vis = tile_visible(tl, R, t)
        d = np.linalg.norm(tl["center"] - eye)
        if strategy == "S1 send everything": sel[k] = 4
        elif strategy == "S2 frustum only" and vis: sel[k] = 4
        elif strategy in ("S3 frustum + distance", "S4 budgeted greedy") and vis: sel[k] = lod_for_distance(d)
    if strategy == "S4 budgeted greedy":
        # start everything visible at LoD0, then upgrade the tile with best "benefit per byte"
        want, sel = sel, {k: 0 for k in sel}
        spent = sum(len(tiles[k]["lods"][0][0]) * BYTES_PER_POINT for k in sel)
        while True:
            best, best_score = None, 0
            for k, lv in sel.items():
                if lv >= want[k]: continue
                extra = (len(tiles[k]["lods"][lv + 1][0]) - len(tiles[k]["lods"][lv][0])) * BYTES_PER_POINT
                d = np.linalg.norm(tiles[k]["center"] - eye)
                score = 1.0 / (max(d, 1) * max(extra, 1))        # close + cheap first
                if spent + extra <= budget and score > best_score:
                    best, best_score, best_extra = k, score, extra
            if best is None: break
            sel[best] += 1; spent += best_extra
    return sel, (R, t)

def render_selection(sel, R, t):
    if not sel:
        return np.ones((H, W, 3))
    out = None
    for lv in sorted(set(sel.values())):                 # coarse LoD -> draw BIGGER points (like Potree)
        ks = [k for k, l in sel.items() if l == lv]
        pts = np.vstack([tiles[k]["lods"][lv][0] for k in ks])
        col = np.vstack([tiles[k]["lods"][lv][1] for k in ks])
        out = c3d.render(pts, col, K, R, t, W, H, point_px=POINT_PX[lv], bg=(0.85, 0.92, 1), out=out)
    return out[0]

def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return 99 if mse == 0 else 10 * np.log10(1 / mse)

strategies = ["S1 send everything", "S2 frustum only", "S3 frustum + distance", "S4 budgeted greedy"]
BUDGET = 0.6e6                                       # bytes per update (~4.8 Mbit/s at 1 update/s)
stats = {s: dict(bytes=[], psnr=[]) for s in strategies}
snap = {}
for i, pose in enumerate(path):
    sel_full, (R, t) = choose("S1 send everything", pose)
    ref = render_selection(sel_full, R, t)
    for s in strategies:
        sel, _ = choose(s, pose, BUDGET)
        b = sum(len(tiles[k]["lods"][lv][0]) for k, lv in sel.items()) * BYTES_PER_POINT
        img = render_selection(sel, R, t)
        stats[s]["bytes"].append(b); stats[s]["psnr"].append(psnr(img, ref))
        if i == 20: snap[s] = (img, sel)
print(f"\n{'strategy':24s} {'avg MB/update':>14s} {'vs S1':>7s} {'avg PSNR (dB)':>14s}")
for s in strategies:
    mb = np.mean(stats[s]["bytes"]) / 1e6
    print(f"{s:24s} {mb:14.2f} {mb / (np.mean(stats[strategies[0]]['bytes']) / 1e6):7.0%} {np.mean(stats[s]['psnr']):14.1f}")
print(f"(S4 budget = {BUDGET/1e6:.1f} MB per update; PSNR 99 = identical to full quality)")

# ---------------- figures --------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
for s in strategies:
    ax[0].plot(np.array(stats[s]["bytes"]) / 1e6, label=s); ax[1].plot(stats[s]["psnr"], label=s)
ax[0].set_ylabel("MB needed at this pose"); ax[1].set_ylabel("image PSNR vs full (dB)")
for a in ax: a.set_xlabel("camera path step"); a.grid(alpha=.3); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "strategies.png", dpi=100); plt.close()

fig, ax = plt.subplots(2, 4, figsize=(20, 8))
pose = path[20]
for j, s in enumerate(strategies):
    img, sel = snap[s]
    ax[0, j].imshow(img); ax[0, j].axis("off"); ax[0, j].set_title(f"{s}\n{stats[s]['bytes'][20]/1e6:.2f} MB, {stats[s]['psnr'][20]:.1f} dB")
    grid = np.full((8, 8), -1.0)
    for (gx, gy), lv in sel.items(): grid[gy, gx] = lv
    a = ax[1, j]; a.imshow(np.ma.masked_less(grid, 0), origin="lower", cmap="viridis", vmin=0, vmax=4, extent=[-20, 20, -20, 20])
    a.plot(pose["x"], pose["y"], "r^", ms=12); a.arrow(pose["x"], pose["y"], 6 * np.cos(pose["yaw"]), 6 * np.sin(pose["yaw"]), color="r", width=0.4)
    a.set_title("tile LoD map (white = not sent)"); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(OUT / "tile_lod_maps.png", dpi=90); plt.close()
print("\nsaved strategies.png and tile_lod_maps.png in", OUT)
