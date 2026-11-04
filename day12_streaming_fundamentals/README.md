# Day 12 — Streaming basics: bitrate, buffer and latency

## Run

- `python bitrate_buffer_latency.py`: raw data rates of 2D/360°/volumetric video, a buffer simulation with stalls (`outputs/buffer_simulation.png`), and the motion-to-photon budget.

Outputs go to `outputs/`. Set `COURSE_NO_VIZ=1` to skip 3D windows.

## Try this

1. Change the segment duration to 2 s and 4 s. Effect on stalls?
2. Change the start-up buffer to 0.5 s and 5 s. Trade-off?
3. Write your own bandwidth trace (e.g. a train ride) and test the three bitrates.

Graded work for this session: see `../ASSIGNMENTS.md`.
