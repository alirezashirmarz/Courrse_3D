# Day 01 — What is 3D, and why do we need it?

## Run

- `python 01_dimensions.py` → `outputs/dimensions.png`: the same object as 1D, 2D, 2.5D and 3D.
- `python 02_parallax_3dof_vs_6dof.py` → `outputs/parallax.png`: rotation-only views vs moving views.
- Open `dof_playground.html` (serve with `python -m http.server 8000`, then http://localhost:8000/dof_playground.html). Press **M** to switch 3DoF ↔ 6DoF.

Outputs go to `outputs/`. Set `COURSE_NO_VIZ=1` to skip 3D windows.

## Try this

1. In the playground, stand behind the orange pillars in 3DoF mode: can you see the ball? Switch to 6DoF and step sideways.
2. In `01_dimensions.py`, move the ball to the side of the hill (`(1.5, 0, 1.2)`) and check whether the depth map now shows both.
3. In `02_parallax...py`, add a view that does BOTH: yaw 30° and step 3 m left. Which DoF changed the image more?

Graded work for this session: see `../ASSIGNMENTS.md`.
