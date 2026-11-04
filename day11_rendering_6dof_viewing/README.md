# Day 11 — The view frustum and a real 6DoF viewer

## Run

- `python frustum_culling.py`: what fraction of the scene each pose can see (`outputs/frustum.png`). It also writes `viewer6dof/scene.ply`.
- `cd viewer6dof && python -m http.server 8000` → http://localhost:8000. Fly around; watch 'Visible in frustum'. Press **● Record pose trace** to save your 6DoF path as CSV (used on Day 16).

Outputs go to `outputs/`. Set `COURSE_NO_VIZ=1` to skip 3D windows.

## Try this

1. Find the pose where the fewest points are visible while the house is still on screen.
2. Change the field-of-view slider: how does the visible fraction change?
3. Record a 30-second trace for Day 16.

Graded work for this session: see `../ASSIGNMENTS.md`.
