# Day 13 — Compressing point clouds (G-PCC and V-PCC ideas)

## Run

- `python compress_pointcloud.py`: quantisation, an octree occupancy coder from scratch, colour quantisation, rate-distortion curve (`outputs/rate_distortion.png`).
- `python vpcc_projection_toy.py`: project the 3D person onto 6 images, compress them with PNG/JPEG, rebuild 3D (`outputs/vpcc_toy.png`).

Outputs go to `outputs/`. Set `COURSE_NO_VIZ=1` to skip 3D windows.

## Try this

1. `pip install DracoPy` and compare Draco with our octree coder on the RD plot.
2. In the V-PCC toy, lower JPEG quality to 30: which errors do you see in 3D?
3. Replace zlib in the octree coder with `lzma`. Does it help?

Graded work for this session: see `../ASSIGNMENTS.md`.
