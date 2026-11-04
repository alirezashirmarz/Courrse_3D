"""
Day 16 — Volumetric video streaming server (DASH-style, for the browser player)
------------------------------------------------------------------------------
Run:   python vv_server.py            then open http://localhost:8001

* Generates a 6-second looping volumetric video (a waving person, 15 fps)
* Encodes it as 1-second SEGMENTS at 3 QUALITY LEVELS (the bitrate ladder)
* Publishes manifest.json (like a DASH .mpd) and serves /seg?level=L&index=k
* All downloads go through a simulated, shared network link (bandwidth + latency)
The browser player (player/index.html) runs the ABR logic and renders in 6DoF.
"""
import argparse, json, struct, sys, threading, time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import course3d as c3d

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8001)
ap.add_argument("--kbps", type=float, default=15000)
ap.add_argument("--latency-ms", type=float, default=60)
args = ap.parse_args()

FPS, SEG_FRAMES, N_SEG = 15, 15, 6
LEVELS = [3_000, 10_000, 25_000]               # points per frame
LO = np.array([-0.6, 0.0, -0.4], np.float32)   # Y-up bounding box origin
STEP = np.float32(2.0 / 65535)

c3d.banner("Day 16 · encoding the volumetric video ladder")
SEGS = {}
for li, n in enumerate(LEVELS):
    for k in range(N_SEG):
        parts = [struct.pack("<I", SEG_FRAMES)]
        for f in range(SEG_FRAMES):
            pts, col = c3d.make_human((k * SEG_FRAMES + f) / FPS, n, rng=np.random.default_rng(f + 100 * k))
            pts = np.c_[pts[:, 0], pts[:, 2], -pts[:, 1]]              # Z-up -> Y-up
            q = np.clip(np.round((pts - LO) / STEP), 0, 65535).astype(np.uint16)
            parts += [struct.pack("<I", len(pts)), q.tobytes(), (col * 255).astype(np.uint8).tobytes()]
        SEGS[(li, k)] = b"".join(parts)
    mbps = np.mean([len(SEGS[(li, k)]) for k in range(N_SEG)]) * 8 / (SEG_FRAMES / FPS) / 1e6
    print(f"  level {li}: {n:6,d} pts/frame -> {mbps:5.1f} Mbit/s")
MANIFEST = json.dumps(dict(
    fps=FPS, segmentDuration=SEG_FRAMES / FPS, segments=N_SEG, loop=True,
    origin=LO.tolist(), step=float(STEP),
    levels=[dict(points=n, bitrate=float(np.mean([len(SEGS[(li, k)]) for k in range(N_SEG)]) * 8 / (SEG_FRAMES / FPS)))
            for li, n in enumerate(LEVELS)])).encode()

class Link:
    def __init__(s, kbps, lat): s.kbps, s.latency, s.lock, s.next_free = kbps, lat / 1000, threading.Lock(), time.time()
    def reserve(s, nbytes):
        with s.lock:
            start = max(time.time(), s.next_free); s.next_free = start + nbytes * 8 / (s.kbps * 1000); return s.next_free
LINK = Link(args.kbps, args.latency_ms)

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw): super().__init__(*a, directory=str(HERE / "player"), **kw)
    def log_message(self, *a): pass
    def _send(self, body, ctype, throttle=True):
        time.sleep(LINK.latency)
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store"); self.end_headers()
        for i in range(0, len(body), 16384):
            part = body[i:i + 16384]
            if throttle: time.sleep(max(0, LINK.reserve(len(part)) - time.time()))
            self.wfile.write(part)
    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass                                       # the browser closed the connection - fine
    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path == "/manifest.json": return self._send(MANIFEST, "application/json", False)
        if u.path == "/seg":
            key = (int(q["level"][0]), int(q["index"][0]) % N_SEG)
            print(f"  -> segment {key[1]} at level {key[0]} ({c3d.human_bytes(len(SEGS[key]))}), link {LINK.kbps/1000:.0f} Mbit/s")
            return self._send(SEGS[key], "application/octet-stream")
        if u.path == "/set":
            LINK.kbps = float(q.get("kbps", [LINK.kbps])[0]); LINK.latency = float(q.get("latency_ms", [LINK.latency * 1000])[0]) / 1000
            return self._send(json.dumps(dict(kbps=LINK.kbps, latency_ms=LINK.latency * 1000)).encode(), "application/json", False)
        return super().do_GET()

print(f"\nOpen http://localhost:{args.port}   (link {args.kbps/1000:.0f} Mbit/s, {args.latency_ms:.0f} ms).  Ctrl+C to stop.")
ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
