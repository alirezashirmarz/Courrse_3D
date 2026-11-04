# Day 16 — Volumetric video, ABR, QoE and pose prediction

## Run

- `python abr_volumetric.py`: bitrate ladder for a volumetric person; fixed vs throughput vs buffer ABR; stalls, switches, QoE (`outputs/abr_comparison.png`).
- `python pose_prediction.py [pose_trace.csv]`: predict where the viewer will be in 50–800 ms.
- `python vv_server.py` then http://localhost:8001: a volumetric video player with live ABR you can walk around in 6DoF.

Outputs go to `outputs/`. Set `COURSE_NO_VIZ=1` to skip 3D windows.

## Try this

1. In the player, set bandwidth to 5 Mbit/s and compare the four ABR choices by their QoE.
2. Run `pose_prediction.py` on YOUR trace from Day 11. Which predictor wins?
3. Design a hybrid ABR (buffer + throughput) and beat BBA's QoE in `abr_volumetric.py`.

Graded work for this session: see `../ASSIGNMENTS.md`.
