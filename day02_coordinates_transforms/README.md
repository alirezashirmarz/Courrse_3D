# Day 02 — Coordinates, transformations and the 6DoF pose

## Run

- `python transforms.py` prints every step (points, rotation, 4×4 matrices, order, inverse, quaternion) and saves `outputs/transforms.png`.

Outputs go to `outputs/`. Set `COURSE_NO_VIZ=1` to skip 3D windows.

## Try this

1. Change the rotation to 45° about X (pitch) and print `R @ [0,0,1]`. Where does 'up' go?
2. Prove numerically that `inv(T)` equals `[[R.T, -R.T @ t],[0,0,0,1]]`.
3. Write `quat_to_R(q)` and check `quat_to_R(R_to_quat(R))` gives back R.

Graded work for this session: see `../ASSIGNMENTS.md`.
