# Day 07 — k-d trees, octrees and Levels of Detail

## Run

- `python kdtree_octree_lod.py`: brute force vs k-d tree timing, an octree written from scratch with occupancy statistics per level, and a 5-level LoD pyramid (`outputs/lod_pyramid.png`, `lod0..4.ply`).

Outputs go to `outputs/`. Set `COURSE_NO_VIZ=1` to skip 3D windows.

## Try this

1. Double the number of points. How do brute-force and k-d tree times scale?
2. Change the LoD representative from centroid to 'first point in the cell'. Visual difference?
3. Estimate how long each LoD takes to download over 4G (20 Mbit/s).

Graded work for this session: see `../ASSIGNMENTS.md`.
