"""
Day 04 — Point clouds up close: attributes, file formats and SIZE
----------------------------------------------------------------
Run:  python formats_and_sizes.py

Streaming is all about bytes.  Here we save the SAME point cloud in
several formats and compare file size + load time.  We also compute
"bytes per point" — the number that decides how much bandwidth a 3D
stream needs.
"""
import sys, time, zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import course3d as c3d

OUT = c3d.out_dir(__file__)
c3d.banner("Day 04 · Point cloud formats & sizes")
P, C, L = c3d.make_scene(200_000)
N = len(P)
print(f"Point cloud: {N:,} points. Attributes: x,y,z (float) + r,g,b (0-255) + class label")

rows = []
def record(name, path, loader):
    t0 = time.perf_counter(); loader(); dt = time.perf_counter() - t0
    size = path.stat().st_size
    rows.append((name, size, size / N, dt))

# 1) ASCII text (like .xyz / .txt / .csv)
p = OUT / "cloud.txt"
np.savetxt(p, np.c_[P, (C * 255).round(), L], fmt=["%.3f"] * 3 + ["%d"] * 4, delimiter=" ")
record("ASCII .txt (x y z r g b c)", p, lambda: np.loadtxt(p))

# 2) PLY ascii vs binary (our tiny writer)
p2 = c3d.write_ply(OUT / "cloud_ascii.ply", P, C, binary=False)
record("PLY ascii", p2, lambda: c3d.read_ply(p2) if c3d.try_open3d() else p2.read_text())
p3 = c3d.write_ply(OUT / "cloud_binary.ply", P, C, binary=True)
record("PLY binary (float32 xyz + uint8 rgb)", p3, lambda: c3d.read_ply(p3) if c3d.try_open3d() else p3.read_bytes())

# 3) raw numpy
p4 = OUT / "cloud.npy"; np.save(p4, np.c_[P, C].astype(np.float32))
record("NumPy .npy (7 x float32)", p4, lambda: np.load(p4))

# 4) quantised: 16-bit integer coordinates (1 mm grid) + 8-bit colour, then zlib
mn = P.min(0); q = np.round((P - mn) / 0.001).astype(np.uint32)
print("Quantised range needs", int(np.ceil(np.log2(q.max() + 1))), "bits per axis at 1 mm precision")
qcoords = q.astype(np.uint16) if q.max() < 65536 else q          # 2 bytes per axis if it fits
blob = zlib.compress(qcoords.tobytes() + (C * 255).astype(np.uint8).tobytes(), 9)
p5 = OUT / "cloud_quant_zlib.bin"; p5.write_bytes(blob)
record("Quantised (1 mm) + zlib", p5, lambda: zlib.decompress(p5.read_bytes()))

# 5) LAS (the LiDAR standard) if laspy is installed
try:
    import laspy
    hdr = laspy.LasHeader(point_format=2, version="1.2"); hdr.scales = [0.001] * 3; hdr.offsets = mn
    las = laspy.LasData(hdr); las.x, las.y, las.z = P.T
    las.red, las.green, las.blue = (C * 65535).astype(np.uint16).T; las.classification = L.astype(np.uint8)
    p6 = OUT / "cloud.las"; las.write(p6)
    record("LAS 1.2 (LiDAR standard)", p6, lambda: laspy.read(p6))
except ImportError:
    print("(laspy not installed -> skipping LAS. pip install laspy to add it)")

print(f"\n{'format':40s} {'size':>12s} {'bytes/point':>12s} {'load time':>10s}")
for name, size, bpp, dt in rows:
    print(f"{name:40s} {c3d.human_bytes(size):>12s} {bpp:12.1f} {dt*1000:8.0f} ms")

print("\nWhat does this mean for STREAMING?")
bpp = rows[[r[0] for r in rows].index("PLY binary (float32 xyz + uint8 rgb)")][1] / N
for pts in [100_000, 1_000_000]:
    for fps in [30]:
        mbps = pts * bpp * 8 * fps / 1e6
        print(f"  {pts:>9,} points x {bpp:.0f} B x {fps} fps = {mbps:,.0f} Mbit/s uncompressed  (a 4K Netflix stream is ~15 Mbit/s)")
