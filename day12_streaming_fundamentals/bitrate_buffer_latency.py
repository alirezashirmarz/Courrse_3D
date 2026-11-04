"""
Day 12 — Streaming fundamentals: bitrate, bandwidth, buffer, latency
-------------------------------------------------------------------
Run:  python bitrate_buffer_latency.py

Part A  How big is 3D video?  2D video vs 360° video vs volumetric video
Part B  Download segments over a fluctuating network: watch the BUFFER fill,
        drain and STALL (the #1 cause of bad Quality of Experience)
Part C  The 6DoF latency budget: motion-to-photon must stay < ~20 ms,
        which is why the client renders locally and streams the CONTENT,
        not the pixels.
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
c3d.banner("Day 12 · Streaming fundamentals")

# ---------------- Part A: how big? ------------------------------------------------
print("[A] Raw (uncompressed) data rates")
def mbps(bits_per_frame, fps):
    return bits_per_frame * fps / 1e6
rows = [
    ("HD video 1920x1080, 24 bit, 30 fps", mbps(1920 * 1080 * 24, 30), 5),
    ("360° video 7680x3840 (8K), 30 fps", mbps(7680 * 3840 * 24, 30), 50),
    ("Volumetric: 100k pts x 15 B, 30 fps", mbps(100_000 * 15 * 8, 30), None),
    ("Volumetric: 1M pts (MPEG 8i test) x 15 B, 30 fps", mbps(1_000_000 * 15 * 8, 30), None),
    ("Static city scan: 1 billion pts x 15 B (one-off)", 1e9 * 15 * 8 / 1e6, None),
]
print(f"    {'content':52s} {'raw Mbit/s':>14s}  {'typical compressed':>18s}")
for name, raw, comp in rows:
    print(f"    {name:52s} {raw:14,.0f}  {str(comp) + ' Mbit/s' if comp else '(Day 13!)':>18s}")
print("    Home broadband ~ 50-300 Mbit/s, 4G ~ 20, 5G ~ 100-500  -> raw 3D does NOT fit.")

# ---------------- Part B: buffer simulation -----------------------------------------
rng = np.random.default_rng(7)
T = 120                                                   # seconds
bw = np.clip(25 + 12 * np.sin(np.arange(T) / 9) + rng.normal(0, 5, T), 2, None)   # Mbit/s trace
bw[60:75] = 4                                             # a tunnel! bandwidth drops
seg_dur = 1.0                                             # 1-second segments (like DASH)

def simulate(bitrate_mbps, start_buffer=2.0, max_buffer=8.0):
    """Download 1 s segments back-to-back; play them. Returns timeline."""
    t, buf, played, stall = 0.0, 0.0, 0.0, 0.0
    times, bufs, stalls = [], [], []
    seg_bits = bitrate_mbps * seg_dur
    playing = False
    while played < T - 10:
        if buf > max_buffer - seg_dur:               # buffer full -> wait
            dt = 0.1
            t += dt; buf -= dt; played += dt
        else:
            need, dl = seg_bits, 0.0                     # download one segment
            while need > 1e-6:                           # (tiny tolerance avoids float round-off loops)
                rate = bw[min(int(t + dl), T - 1)]
                step = min(0.05, need / rate); need -= rate * step; dl += step
            if playing:
                drained = min(buf, dl); buf -= drained; played += drained
                if dl > drained:
                    stall += dl - drained
            t += dl; buf += seg_dur
            if not playing and buf >= start_buffer:
                playing = True
        times.append(t); bufs.append(buf); stalls.append(stall)
    return np.array(times), np.array(bufs), stall

fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax[0].plot(bw, color="#2f6fed"); ax[0].set_ylabel("bandwidth (Mbit/s)"); ax[0].set_title("Network bandwidth trace (tunnel at 60-75 s)")
print("\n[B] Fixed-bitrate streaming over the trace:")
for br, col in [(10, "#2a9d8f"), (20, "#f08a24"), (30, "#d64545")]:
    ti, bu, st = simulate(br)
    ax[0].axhline(br, color=col, ls="--", lw=1)
    ax[1].plot(ti, bu, color=col, label=f"{br} Mbit/s content -> stall {st:.1f} s")
    print(f"    content at {br:2d} Mbit/s  ->  total stall time {st:5.1f} s")
ax[1].set_ylabel("buffer (s of content)"); ax[1].set_xlabel("time (s)"); ax[1].legend(); ax[1].axhline(0, color="k", lw=.5)
plt.tight_layout(); plt.savefig(OUT / "buffer_simulation.png", dpi=100); plt.close()
print("    -> higher quality = more stalls when the network dips. Adaptive bitrate (ABR) fixes this (Day 16).")

# ---------------- Part C: latency budget ------------------------------------------------
print("\n[C] Motion-to-photon (MTP) budget for 6DoF (target < 20 ms)")
budget = [("head-pose sensor sampling", 2), ("pose -> renderer", 1), ("render frame (90 Hz)", 11), ("display scan-out", 5)]
for n, ms in budget:
    print(f"    {n:30s} {ms:3d} ms")
print(f"    {'TOTAL (local rendering)':30s} {sum(m for _, m in budget):3d} ms  OK")
rtt = 40
print(f"    + a network round-trip ({rtt} ms) if the SERVER rendered the pixels -> {sum(m for _, m in budget) + rtt} ms  -> motion sickness!")
print("    => 6DoF streaming sends 3D CONTENT ahead of time; the headset renders locally at 90 Hz.")
print("\nsaved", OUT / "buffer_simulation.png")
