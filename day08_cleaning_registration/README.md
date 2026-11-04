# Day 08 — Cleaning and registration (ICP finds a 6DoF pose)

## Run

- `python clean_and_register.py`: noise and outliers → voxel downsample → statistical outlier removal → normals → ICP recovers a secret 6DoF pose. Four 3D windows show each stage.

Outputs go to `outputs/`. Set `COURSE_NO_VIZ=1` to skip 3D windows.

## Try this

1. Switch `TransformationEstimationPointToPlane` to `PointToPoint`. Compare iterations and final error.
2. Increase the secret rotation to 45°. When does ICP fail? Why does it need a good initial guess?
3. Change `std_ratio` from 2.0 to 0.5. What good points get removed?

Graded work for this session: see `../ASSIGNMENTS.md`.
