"""
Day 16 — Volumetric video streaming with Adaptive Bitrate (ABR) and QoE
----------------------------------------------------------------------
Run:  python abr_volumetric.py

A volumetric video = a sequence of point clouds (a 3D "movie" you can walk
around in 6DoF).  We:
  1. generate a waving-human sequence and encode each 1-second SEGMENT at
     3 quality levels (like a DASH "bitrate ladder"),
  2. stream it over a fluctuating network with 3 ABR algorithms:
        fixed-high    always the best quality
        throughput    pick the highest level below the measured throughput
        buffer (BBA)  pick quality from how full the buffer is
  3. score them with the QoE model used in ABR research (MPC / Pensieve):
        QoE = sum of bitrates  -  mu x stall seconds  -  sum of |bitrate changes|
        (mu = the highest bitrate: 1 s of stall hurts as much as losing 1 s of top quality)
     reported per segment so it reads like "useful Mbit/s".
"""
import sys, zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import course3d as c3d

OUT = c3d.out_dir(__file__)
c3d.banner("Day 16 · Volumetric video + ABR")
FPS, SEG = 15, 1.0                       # 15 frames/s, 1-second segments
LEVELS = [4_000, 12_000, 40_000]         # points per frame for the 3 quality levels
N_SEG = 10                               # 10 s of content (looped during streaming)

# ---------------- 1) encode the bitrate ladder ---------------------------------------------
def encode_frame(pts, col, bits=10):
    lo = np.array([-0.6, -0.4, -0.05]); step = 2.0 / (2**bits)
    q = np.clip(np.round((pts - lo) / step), 0, 2**bits - 1).astype(np.uint16)
    return zlib.compress(q.tobytes() + (col * 31).astype(np.uint8).tobytes(), 6)

seg_bytes = np.zeros((N_SEG, len(LEVELS)))
for s in range(N_SEG):
    for li, n in enumerate(LEVELS):
        tot = 0
        for f in range(0, int(FPS * SEG), 3):              # encode every 3rd frame, scale (speed)
            pts, col = c3d.make_human((s * SEG) + f / FPS, n)
            tot += len(encode_frame(pts, col)) * 3
        seg_bytes[s, li] = tot
ladder_mbps = seg_bytes.mean(0) * 8 / SEG / 1e6
print("Bitrate ladder (like a DASH manifest):")
for li, n in enumerate(LEVELS):
    print(f"   level {li}: {n:6,d} points/frame -> {ladder_mbps[li]:6.2f} Mbit/s")

# ---------------- 2) network trace -----------------------------------------------------------
rng = np.random.default_rng(11)
T = 180
scale = ladder_mbps[-1]
bw = np.clip(scale * (0.75 + 0.45 * np.sin(np.arange(T) / 12)) + rng.normal(0, scale * 0.15, T), scale * 0.08, None)
bw[60:100] *= 0.2                                      # congestion episode (e.g. a crowded cell)

def stream(abr, max_buf=6.0, start_buf=1.0):
    t, buf, playing = 0.0, 0.0, False
    stall, chosen, buf_log, thr_hist = 0.0, [], [], []
    for k in range(60):                                # 60 segments
        level = abr(buf, thr_hist)
        size_bits = seg_bytes[k % N_SEG, level] * 8
        dl, left = 0.0, size_bits
        while left > 1:                                # (> 1 bit: avoids float round-off loops)
            rate = bw[min(int(t + dl), T - 1)] * 1e6
            step = min(0.05, left / rate); left -= rate * step; dl += step
        if playing:
            if dl > buf: stall += dl - buf; buf = 0
            else: buf -= dl
        t += dl; buf += SEG
        thr_hist.append(size_bits / dl / 1e6)
        chosen.append(level); buf_log.append((t, buf))
        if not playing and buf >= start_buf: playing = True
        if buf > max_buf:                              # wait (keep playing) until room in buffer
            wait = buf - max_buf; t += wait; buf = max_buf
    switches = int(np.sum(np.diff(chosen) != 0))
    quality = np.mean(chosen)
    rates = ladder_mbps[chosen]
    mu = ladder_mbps[-1]
    qoe = (rates.sum() - mu * stall - np.abs(np.diff(rates)).sum()) / len(chosen)
    return dict(levels=chosen, buf=np.array(buf_log), stall=stall, switches=switches, quality=quality, qoe=qoe)

def abr_fixed(buf, hist): return len(LEVELS) - 1
def abr_throughput(buf, hist):
    if not hist: return 0
    est = 0.8 * np.mean(hist[-3:])                      # safety margin 20%
    ok = [i for i, r in enumerate(ladder_mbps) if r <= est]
    return max(ok) if ok else 0
def abr_buffer(buf, hist):                               # BBA: reservoir 1.5 s, cushion up to 5 s
    if buf < 1.5: return 0
    if buf > 4.5: return len(LEVELS) - 1
    return 1

results = {n: stream(f) for n, f in [("fixed-high", abr_fixed), ("throughput-based", abr_throughput), ("buffer-based (BBA)", abr_buffer)]}
print(f"\n{'ABR':20s} {'avg level':>9s} {'stall (s)':>10s} {'switches':>9s} {'QoE/segment':>12s}")
for n, r in results.items():
    print(f"{n:20s} {r['quality']:9.2f} {r['stall']:10.1f} {r['switches']:9d} {r['qoe']:12.2f}")

fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
ax[0].plot(bw, color="#2f6fed"); [ax[0].axhline(r, ls="--", color="gray", lw=.8) for r in ladder_mbps]
ax[0].set_ylabel("bandwidth (Mbit/s)"); ax[0].set_title("network trace (dashed = the 3 quality levels)")
for (n, r), c in zip(results.items(), ["#d64545", "#f08a24", "#2a9d8f"]):
    tt = r["buf"][:, 0]
    ax[1].step(tt, r["levels"], where="post", color=c, label=n)
    ax[2].plot(tt, r["buf"][:, 1], color=c, label=f"{n}: stall {r['stall']:.1f}s, QoE {r['qoe']:.2f}")
ax[1].set_ylabel("quality level"); ax[1].set_yticks([0, 1, 2]); ax[1].legend(fontsize=8)
ax[2].set_ylabel("buffer (s)"); ax[2].set_xlabel("time (s)"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "abr_comparison.png", dpi=100); plt.close()

# a filmstrip of the volumetric video at the 3 levels
fig, ax = plt.subplots(3, 5, figsize=(15, 10))
K = c3d.intrinsics(240, 320, 45); R, t = c3d.look_at((1.5, -3.2, 1.3), (0, 0, 0.9))
for li, n in enumerate(LEVELS):
    for j, tt in enumerate(np.linspace(0, 1, 5)):
        pts, col = c3d.make_human(tt, n)
        ax[li, j].imshow(c3d.render(pts, col, K, R, t, 240, 320, point_px=[4, 3, 2][li])[0]); ax[li, j].axis("off")
        if j == 0: ax[li, j].set_title(f"level {li}: {n:,} pts/frame ({ladder_mbps[li]:.1f} Mbit/s)", loc="left")
plt.tight_layout(); plt.savefig(OUT / "volumetric_filmstrip.png", dpi=90); plt.close()
print("\nsaved abr_comparison.png and volumetric_filmstrip.png in", OUT)
