# Day 15 — A live 6DoF streaming system

## Run

- `python server.py` (packages tiles × LoDs, serves a manifest and chunks through a simulated network link).
- Open http://localhost:8000. Fly around. Watch the tile map, the throughput and 'visible tiles at target quality'. Switch strategy and move the bandwidth/latency sliders.

Outputs go to `outputs/`. Set `COURSE_NO_VIZ=1` to skip 3D windows.

## Try this

1. Set bandwidth to 2 Mbit/s: compare *naive* and *viewport-adaptive* — how long until the view looks good?
2. Set latency to 300 ms and walk fast. What breaks?
3. In `client/index.html`, change `MAX_INFLIGHT` from 4 to 1 and 16.

Graded work for this session: see `../ASSIGNMENTS.md`.
