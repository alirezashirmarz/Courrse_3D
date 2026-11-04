# Day 04 — Point clouds up close: attributes, formats and bytes

## Run

- `python formats_and_sizes.py` saves the same 190k-point cloud as TXT, PLY ascii/binary, NPY and quantised+zlib, and prints size, bytes/point and load time.

Outputs go to `outputs/`. Set `COURSE_NO_VIZ=1` to skip 3D windows.

## Try this

1. `pip install laspy` and rerun: where does LAS land in the table?
2. Change the quantisation step from 1 mm to 1 cm. How many bytes/point now?
3. Compute how many points per frame you could stream at 30 fps over a 25 Mbit/s link for each format.

Graded work for this session: see `../ASSIGNMENTS.md`.
