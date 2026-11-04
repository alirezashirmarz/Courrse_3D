# Day 10 — Meshes, mesh LoDs and Gaussian splats (book Ch. 10, 18)

## Run

- `python mesh_lod_and_splats.py`: Poisson meshing, quadric decimation into 5 LoDs with file sizes, and a tiny Gaussian-splat renderer (`outputs/gaussian_splats.png`).

Outputs go to `outputs/`. Set `COURSE_NO_VIZ=1` to skip 3D windows.

## Try this

1. Change Poisson `depth` 6 → 10. Triangles? Time? Detail?
2. Add a 6th mesh LoD with 50 triangles. At what screen size would you use it?
3. In the splat renderer, make the Gaussian size proportional to local point spacing (use a k-d tree).

Graded work for this session: see `../ASSIGNMENTS.md`.
