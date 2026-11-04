# Day 09 — From geometry to meaning: features, RANSAC, DBSCAN

## Run

- `python segment_scene.py`: PCA features per point, RANSAC ground plane from scratch, DBSCAN objects, precision/recall of the ground, `outputs/segmentation.png`.

Outputs go to `outputs/`. Set `COURSE_NO_VIZ=1` to skip 3D windows.

## Try this

1. Change DBSCAN `eps` to 0.2 and 1.5. Why do trees split or merge?
2. Run RANSAC a second time on the non-ground points: what plane does it find?
3. Use verticality > 0.8 to extract walls only and export them.

Graded work for this session: see `../ASSIGNMENTS.md`.
