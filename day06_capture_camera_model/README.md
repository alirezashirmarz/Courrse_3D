# Day 06 — How 3D is captured: the pinhole camera and RGB-D

## Run

- `python pinhole_and_rgbd.py`: projects a point by hand, simulates an RGB-D sensor, back-projects to a cloud, shows the damage of a wrong focal length, and fuses two views (`outputs/*.ply`, `camera_model.png`).

Outputs go to `outputs/`. Set `COURSE_NO_VIZ=1` to skip 3D windows.

## Try this

1. Change `fov_deg` to 30 and 100. What happens to `fx`? To the point density?
2. Add Gaussian noise (σ = 1% of depth) to the depth image before back-projection, like a real sensor.
3. Add a third view from behind the house and measure how many new points it contributes.

Graded work for this session: see `../ASSIGNMENTS.md`.
