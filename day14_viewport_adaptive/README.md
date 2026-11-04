# Day 14 — Viewport-adaptive 6DoF streaming

## Run

- `python viewport_adaptive.py`: tiles × LoDs; a camera path; 4 strategies (everything / frustum / frustum+distance / budgeted greedy) compared on MB and image PSNR (`outputs/strategies.png`, `tile_lod_maps.png`).

Outputs go to `outputs/`. Set `COURSE_NO_VIZ=1` to skip 3D windows.

## Try this

1. Change `BUDGET`. Plot PSNR vs budget for S4.
2. Make the tiles 2.5 m or 10 m. What is the overhead/precision trade-off?
3. Add a 'margin' to the frustum test (fetch tiles slightly outside the view). When does it help?

Graded work for this session: see `../ASSIGNMENTS.md`.
