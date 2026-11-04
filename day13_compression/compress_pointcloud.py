"""
Day 13 — Compressing point clouds (the idea behind MPEG G-PCC and Draco)
-----------------------------------------------------------------------
Run:  python compress_pointcloud.py

Part A  QUANTISATION: snap coordinates to a grid of 2^bits cells.
        Fewer bits -> smaller, but less precise.  We measure the error
        with the standard metric "D1 point-to-point PSNR".
Part B  OCTREE OCCUPANCY CODING from scratch: instead of sending x,y,z,
        send 1 byte per octree node saying which of its 8 children are
        occupied.  Empty space costs nothing.  (This is G-PCC's geometry coder.)
Part C  Colour: 8-bit RGB -> fewer bits per channel.
Part D  Rate-distortion curve: bits per point vs quality.
Optional: if `pip install DracoPy` works, compare with Google Draco.
"""
import sys, zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import course3d as c3d

OUT = c3d.out_dir(__file__)
c3d.banner("Day 13 · Point cloud compression")
P, C, _ = c3d.make_house(60_000)
N = len(P)
raw_bytes = N * 15
print(f"Input: {N:,} points, raw {c3d.human_bytes(raw_bytes)} (12 B xyz float32 + 3 B rgb) = 120 bits/point")

def quantize(points, bits):
    lo = points.min(0); span = np.ptp(points, axis=0).max()
    step = span / (2**bits - 1)
    q = np.round((points - lo) / step).astype(np.int64)
    q = np.unique(q, axis=0)                     # duplicate points merge!
    return q, lo, step

def d1_psnr(ref, rec):
    """MPEG D1 (point-to-point) PSNR, symmetric. Peak = bounding-box diagonal."""
    d_ab, _ = cKDTree(rec).query(ref); d_ba, _ = cKDTree(ref).query(rec)
    mse = max((d_ab**2).mean(), (d_ba**2).mean())
    peak = np.linalg.norm(np.ptp(ref, axis=0))
    return 10 * np.log10(peak**2 / mse) if mse > 0 else np.inf

# ---------------- Part A: quantisation ---------------------------------------------
print("\n[A] Quantisation")
print(f"    {'bits':>4s} {'grid step':>10s} {'points kept':>12s} {'D1 PSNR':>9s}")
for bits in [6, 8, 10, 12, 14]:
    q, lo, step = quantize(P, bits)
    rec = q * step + lo
    print(f"    {bits:4d} {step*100:8.2f} cm {len(q):12,d} {d1_psnr(P, rec):7.1f} dB")

# ---------------- Part B: octree occupancy coding -------------------------------------------
def octree_encode(q, depth):
    """Breadth-first octree occupancy coding of integer points in [0, 2^depth)."""
    codes = bytearray()
    nodes = [q]                                   # each node = array of its points
    for level in range(depth):
        shift = depth - 1 - level
        next_nodes = []
        for pts in nodes:
            child = (((pts[:, 0] >> shift) & 1) << 2) | (((pts[:, 1] >> shift) & 1) << 1) | ((pts[:, 2] >> shift) & 1)
            byte = 0
            for c in range(8):
                sel = pts[child == c]
                if len(sel):
                    byte |= 1 << c
                    next_nodes.append(sel)
            codes.append(byte)
        nodes = next_nodes
    return bytes(codes)

def octree_decode(codes, depth):
    cells = [np.zeros(3, np.int64)]
    pos = 0
    for level in range(depth):
        shift = depth - 1 - level
        nxt = []
        for base in cells:
            byte = codes[pos]; pos += 1
            for c in range(8):
                if byte >> c & 1:
                    nxt.append(base | (np.array([(c >> 2) & 1, (c >> 1) & 1, c & 1]) << shift))
        cells = nxt
    return np.array(cells)

print("\n[B] Octree occupancy coding (geometry only)")
results = []
for bits in [6, 8, 10, 11]:
    q, lo, step = quantize(P, bits)
    codes = octree_encode(q, bits)
    packed = zlib.compress(codes, 9)                     # stand-in for G-PCC's arithmetic coder
    dec = octree_decode(codes, bits)
    lossless = len(dec) == len(q)
    rec = dec * step + lo
    bpp = len(packed) * 8 / N
    psnr = d1_psnr(P, rec)
    results.append((bits, bpp, psnr, len(packed)))
    print(f"    {bits:2d}-bit octree: {len(codes):7,d} occupancy bytes -> zlib {c3d.human_bytes(len(packed)):>9s} = {bpp:5.2f} bits/point,"
          f" D1 PSNR {psnr:5.1f} dB, decoded OK: {lossless}")
print(f"    compare: raw float xyz = 96 bits/point;  zlib on raw floats = {len(zlib.compress(P.astype(np.float32).tobytes(), 9)) * 8 / N:.1f} bits/point")

# ---------------- Part C: colour ------------------------------------------------------------
print("\n[C] Colour quantisation (with realistic sensor colour noise added)")
Cn = np.clip(C + np.random.default_rng(0).normal(0, 0.03, C.shape), 0, 1)
for cb in [8, 5, 3]:
    cq = np.round(Cn * (2**cb - 1)).astype(np.uint8)
    err = np.abs(cq / (2**cb - 1) - Cn).mean() * 255
    print(f"    {cb} bits/channel -> {3 * cb} bits/point colour, mean error {err:4.1f} / 255,"
          f" zlib -> {len(zlib.compress(cq.tobytes(), 9)) * 8 / N:.2f} bits/point")

# ---------------- Part D: rate-distortion -----------------------------------------------------
draco = None
try:
    import DracoPy
    draco = []
    for qb in [6, 8, 10, 11, 12, 14]:
        blob = DracoPy.encode(P.astype(np.float32), quantization_bits=qb, compression_level=7)
        rec = np.asarray(DracoPy.decode(blob).points)
        draco.append((len(blob) * 8 / N, d1_psnr(P, rec)))
    print("\n[D] Draco:", ", ".join(f"{b:.2f} bpp/{p:.1f} dB" for b, p in draco))
except Exception:
    print("\n[D] (DracoPy not installed — `pip install DracoPy` to compare with Google Draco)")

plt.figure(figsize=(7, 4.5))
plt.plot([r[1] for r in results], [r[2] for r in results], "o-", label="our octree + zlib")
for r in results:
    plt.annotate(f"{r[0]}-bit", (r[1], r[2]), textcoords="offset points", xytext=(5, -12))
if draco:
    plt.plot(*zip(*draco), "s--", label="Google Draco")
plt.xlabel("bits per point (geometry)"); plt.ylabel("D1 PSNR (dB)  — higher is better")
plt.title("Rate–distortion: every codec lives on a curve like this"); plt.grid(alpha=.3); plt.legend()
plt.tight_layout(); plt.savefig(OUT / "rate_distortion.png", dpi=110); plt.close()

# visual comparison
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
K = c3d.intrinsics(400, 300, 50); R, t = c3d.look_at((9, -11, 6), (0, 0, 2))
for a, bits in zip(ax, [5, 6, 8, 11]):
    q, lo, step = quantize(P, bits)
    rec = q * step + lo
    _, idx = cKDTree(P).query(rec)
    img, _ = c3d.render(rec, C[idx], K, R, t, 400, 300, point_px=max(2, 14 - 2 * bits + 6))
    a.imshow(img); a.axis("off"); a.set_title(f"{bits}-bit geometry: {len(q):,} pts, {d1_psnr(P, rec):.0f} dB")
plt.tight_layout(); plt.savefig(OUT / "quantisation_visual.png", dpi=100); plt.close()
print("\nsaved rate_distortion.png and quantisation_visual.png in", OUT)
print("\nReal codecs:  MPEG G-PCC = octree + context arithmetic coding (static scans, LiDAR)")
print("              MPEG V-PCC = project to 2D images + HEVC video codec (dynamic humans) -> see vpcc_projection_toy.py")
