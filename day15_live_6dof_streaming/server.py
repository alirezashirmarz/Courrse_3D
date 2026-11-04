"""
Day 15 — A real (tiny) 6DoF streaming server
--------------------------------------------
Run:   python server.py              (then open http://localhost:8000)
       python server.py --kbps 5000  (simulate a slow 5 Mbit/s network)

What it does
  * builds a scene, cuts it into 5 m tiles, each with 5 Levels of Detail
  * encodes every (tile, LoD) as a compact binary chunk:
        16-bit quantised xyz + 8-bit rgb = 9 bytes/point (Day 13 in action)
  * publishes a MANIFEST (manifest.json) that lists tiles, bounding spheres
    and LoD sizes — the 3D equivalent of a DASH .mpd playlist
  * serves chunks over plain HTTP through a SIMULATED NETWORK LINK with a
    bandwidth limit and latency that all requests share (like your Wi-Fi)

The browser client (client/index.html) decides WHAT to ask for, based on
the viewer's 6DoF pose.  The server is "dumb" — exactly like real
DASH / 3D-Tiles / Potree streaming.
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
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--kbps", type=float, default=20000, help="shared link bandwidth in kbit/s")
ap.add_argument("--latency-ms", type=float, default=40, help="one-way request latency")
ap.add_argument("--points", type=int, default=800_000)
args = ap.parse_args()

# ----------------------------------------------------------------------------------------
# 1) Prepare content: tiles x LoDs  (this is the "encoding / packaging" step)
# ----------------------------------------------------------------------------------------
c3d.banner("Day 15 · packaging the scene into tiles and LoDs")
P, C, _ = c3d.make_scene(args.points, extent=20)
P = np.c_[P[:, 0], P[:, 2], -P[:, 1]]          # Z-up -> Y-up (three.js convention)
TILE = 5.0
LOD_CELLS = [1.2, 0.6, 0.3, 0.12, 0.0]
gx = np.floor((P[:, 0] + 20) / TILE).astype(int); gz = np.floor((P[:, 2] + 20) / TILE).astype(int)
CHUNKS, manifest_tiles = {}, []

def encode(pts, col):
    lo = pts.min(0); span = max(np.ptp(pts, axis=0).max(), 1e-6)
    step = span / 65535
    q = np.round((pts - lo) / step).astype(np.uint16)
    header = struct.pack("<I4f", len(pts), *lo.astype(np.float32), step)
    return header + q.tobytes() + (col * 255).astype(np.uint8).tobytes()

for key in sorted(set(zip(gx, gz))):
    m = (gx == key[0]) & (gz == key[1])
    pts, col = P[m], C[m]
    tid = f"{key[0]}_{key[1]}"
    lods = []
    for li, cell in enumerate(LOD_CELLS):
        if cell:
            _, first = np.unique(np.floor(pts / cell).astype(np.int64), axis=0, return_index=True)
            lp, lc = pts[first], col[first]
        else:
            lp, lc = pts, col
        blob = encode(lp, lc)
        CHUNKS[(tid, li)] = blob
        lods.append(dict(points=int(len(lp)), bytes=len(blob)))
    lo, hi = pts.min(0), pts.max(0)
    manifest_tiles.append(dict(id=tid, center=((lo + hi) / 2).round(3).tolist(),
                               radius=float(np.linalg.norm(hi - lo) / 2), lods=lods))
MANIFEST = json.dumps(dict(version=1, tileSize=TILE, lodCells=LOD_CELLS, upAxis="Y",
                           bytesPerPoint=9, tiles=manifest_tiles)).encode()
total = sum(len(b) for (t, l), b in CHUNKS.items() if l == len(LOD_CELLS) - 1)
print(f"{len(manifest_tiles)} tiles x {len(LOD_CELLS)} LoDs. Full-quality scene = {c3d.human_bytes(total)}"
      f" ({len(P):,} points). Manifest = {len(MANIFEST):,} bytes")

# ----------------------------------------------------------------------------------------
# 2) A shared, rate-limited network link
# ----------------------------------------------------------------------------------------
class Link:
    def __init__(self, kbps, latency_ms):
        self.kbps, self.latency = kbps, latency_ms / 1000
        self.lock = threading.Lock(); self.next_free = time.time()
        self.sent = 0
    def reserve(self, nbytes):
        with self.lock:
            start = max(time.time(), self.next_free)
            self.next_free = start + nbytes * 8 / (self.kbps * 1000)
            self.sent += nbytes
            return self.next_free
LINK = Link(args.kbps, args.latency_ms)

# ----------------------------------------------------------------------------------------
# 3) HTTP endpoints
# ----------------------------------------------------------------------------------------
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(HERE / "client"), **kw)
    def log_message(self, fmt, *a):
        pass                                           # quiet; we print our own log
    def _send(self, body, ctype, throttle=True):
        time.sleep(LINK.latency)                       # request travels to the server
        self.send_response(200)
        self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store"); self.end_headers()
        CH = 16384
        for i in range(0, len(body), CH):
            part = body[i:i + CH]
            if throttle:
                done_at = LINK.reserve(len(part))
                time.sleep(max(0, done_at - time.time()))
            self.wfile.write(part)
    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass                                       # the browser closed the connection - fine
    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path == "/manifest.json":
            return self._send(MANIFEST, "application/json", throttle=False)
        if u.path == "/tile":
            key = (q["id"][0], int(q["lod"][0]))
            if key not in CHUNKS:
                return self.send_error(404)
            print(f"  -> tile {key[0]:>5s} LoD{key[1]}  {c3d.human_bytes(len(CHUNKS[key])):>9s}   (link {LINK.kbps/1000:.1f} Mbit/s)")
            return self._send(CHUNKS[key], "application/octet-stream")
        if u.path == "/set":                              # change bandwidth live from the client UI
            LINK.kbps = float(q.get("kbps", [LINK.kbps])[0])
            LINK.latency = float(q.get("latency_ms", [LINK.latency * 1000])[0]) / 1000
            return self._send(json.dumps(dict(kbps=LINK.kbps, latency_ms=LINK.latency * 1000)).encode(), "application/json", False)
        return super().do_GET()

print(f"\nLink: {args.kbps/1000:.1f} Mbit/s, {args.latency_ms:.0f} ms latency.")
print(f"Open  http://localhost:{args.port}  in Chrome/Firefox/Edge.   Ctrl+C to stop.\n")
ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
