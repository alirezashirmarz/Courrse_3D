# Day 05 — One object, many representations (book Ch. 4)

## Run

- `python representations.py`: torus as mesh → points → voxels → depth map → points. Prints sizes; saves `outputs/representations.png`; opens a 3D window with all three.

Outputs go to `outputs/`. Set `COURSE_NO_VIZ=1` to skip 3D windows.

## Try this

1. Replace the torus with `create_sphere` or `create_box`. Which representation stores a box most compactly?
2. Try voxel sizes 0.5 and 0.02. Plot number of voxels vs voxel size (log-log). What slope do you expect for a surface?
3. Render the depth map from the SIDE instead of the top. What is missing now?

Graded work for this session: see `../ASSIGNMENTS.md`.
