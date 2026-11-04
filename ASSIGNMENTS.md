# Assignments — From 3D Basics to 6DoF Streaming

**How the work is organised**

| Type | When | Weight | Format |
|---|---|---|---|
| Weekly labs L1–L8 (one per two sessions) | due before the next pair of sessions | 40 % (8 × 5 %) | short notebook or script + ½-page answers |
| Mini-projects P1–P3 | after Sessions 6, 10, 14 | 30 % (3 × 10 %) | code + 2-page report with figures |
| Capstone | Session 16 (demo) + 1 week | 30 % | working demo + 4-page report + 5-min talk |

**Rules for every submission**
1. The code must run with `python your_script.py` from inside the course folder.
2. Every number in your report must come from your code; include the figure or printout.
3. Write the *why*: one sentence explaining each result is worth more than three plots.

---

## Weekly labs

### L1 — Dimensions and degrees of freedom (Sessions 1–2)
1. Using `day01_why_3d/01_dimensions.py`, build a scene where a 2.5D depth map **cannot** represent an object but a 3D point cloud can. Explain why in two sentences.
2. In `dof_playground.html`, find one place where 3DoF and 6DoF give clearly different experiences. Add two screenshots.
3. Write `pose_matrix(x, y, z, yaw, pitch, roll)` and its inverse **without** `np.linalg.inv`. Test it on 1,000 random poses (maximum error < 1e-9).
4. Show with numbers that rotate-then-translate ≠ translate-then-rotate.

*Grading (5 pts):* 1 pt each for Q1–Q2; 2 pts for Q3 (correct + tested); 1 pt for Q4.

### L2 — First program and formats (Sessions 3–4)
1. Load the scene file, compute the bounding box, the mean height per class and the 95th percentile of height. Export each class to its own `.ply`.
2. Fill in the table: format → bytes/point → load time for 100k, 1M and 5M points (scale `make_scene`). Plot size vs number of points.
3. For a 25 Mbit/s link at 30 fps: how many points per frame can you send with each format? Which is the best format for streaming, and why?

*Grading:* 2 / 2 / 1.

### L3 — Representations and capture (Sessions 5–6)
1. Convert one object along the chain mesh → points → voxels → points → mesh (Poisson). Measure what is lost at each step (use the Chamfer distance: the mean nearest-neighbour distance in both directions).
2. With `pinhole_and_rgbd.py`, plot the reconstructed house width against focal-length error from −50 % to +50 %.
3. Fuse 4 RGB-D views around the house. How many views are needed for 95 % surface coverage? (Use voxel occupancy compared with the ground truth.)

*Grading:* 2 / 1.5 / 1.5.

### L4 — Structures and cleaning (Sessions 7–8)
1. Implement `octree_query_box(root, lo, hi)` that returns the indices of points inside a box, using your octree. Compare its speed with a brute-force NumPy mask.
2. Build an LoD pyramid and report, for each LoD, the mean distance between a full-resolution point and its nearest LoD point (the geometric error) against its size in bytes.
3. ICP stress test: for initial rotations of 5°, 15°, 30°, 45° and 60°, run point-to-point and point-to-plane ICP 20 times each with random axes. Plot the success rate.

*Grading:* 2 / 1.5 / 1.5.

### L5 — Understanding and surfaces (Sessions 9–10)
1. Build a small rule-based classifier (ground / wall / roof / tree / other) from PCA features and height. Report precision and recall per class using the true labels.
2. Make 5 mesh LoDs of the house (Poisson, then decimation). For each, render the image from 5 m, 15 m and 40 m and compute PSNR against LoD0. Which LoD is "good enough" at each distance?
3. Briefly explain (≤ 150 words) why Gaussian splats are larger per primitive than points but can still be attractive for streaming.

*Grading:* 2 / 2 / 1.

### L6 — Viewing and streaming basics (Sessions 11–12)
1. Record 3 pose traces in the 6DoF viewer: *walking*, *looking around while standing*, *flying fast*. For each, plot the visible-points fraction over time.
2. Extend the buffer simulator with a simple ABR (choose 10/20/30 Mbit/s each segment). Beat the best fixed bitrate on "average bitrate − 10 × stall seconds".
3. Measure the real round-trip time to 3 servers (`ping`) and explain whether server-side rendering would be acceptable for 6DoF.

*Grading:* 2 / 2 / 1.

### L7 — Compression (Session 13)
1. Add **colour** to the octree coder: send colours in octree (breadth-first) order, quantised to 5 bits per channel, then zlib. Report total bits/point.
2. Plot the rate-distortion curve (D1 PSNR vs bits/point) for 6 to 12 bits. If you can, add Draco (`pip install DracoPy`).
3. In the V-PCC toy, add a **second depth layer** (the farthest point per pixel). How many more points are reconstructed?

*Grading:* 2 / 2 / 1.

### L8 — Adaptive streaming (Sessions 14–15)
1. In `viewport_adaptive.py`, implement **S5 = predicted frustum**: choose tiles using the pose 300 ms in the future (constant-velocity prediction). Compare it with S3 when the request latency is 300 ms.
2. In the live client (`client/index.html`), add a HUD counter **"time to first full view"**: the seconds from Restart until all visible tiles reach their target LoD. Measure it for both strategies at 2, 10 and 50 Mbit/s.
3. Explain one weakness of the throughput estimator in the client and propose a fix.

*Grading:* 2 / 2 / 1.

---

## Mini-projects

### P1 — "Scan to cloud" (after Session 6)
Capture a real object with a phone (photogrammetry app, LiDAR-equipped phone, or a sequence of photos processed in Meshroom/RealityScan/Polycam), **or** use a public RGB-D frame (for example TUM RGB-D or the Open3D sample data).
Deliver: the cleaned point cloud (`.ply`), its statistics (points, bounding box, bytes in 3 formats), and a reconstruction of the camera intrinsics you used.
**Report (2 pages):** capture method, problems met (reflections, holes, scale), and how much data it is.

| Criterion | Points |
|---|---|
| Working pipeline and a clean cloud | 4 |
| Correct statistics and format comparison | 3 |
| Discussion of capture artefacts | 3 |

### P2 — "LoD builder" (after Session 10)
Write `build_lods.py` that takes **any** `.ply` and produces an octree LoD pyramid (points) **and** a mesh LoD pyramid, plus a `manifest.json` listing every LoD with its point/triangle count, bytes and geometric error.
**Report:** curves of size vs error for both pyramids, and a recommendation of which LoD to show at 2 m, 10 m and 50 m.

| Criterion | Points |
|---|---|
| Works on 2 different inputs, including your P1 object | 4 |
| Correct, readable manifest | 2 |
| Error metric and curves | 2 |
| Recommendation, justified | 2 |

### P3 — "Smarter tiles" (after Session 14)
Improve the S4 budgeted greedy strategy. Ideas: an importance score from semantics (Day 9: people and buildings above ground), screen-space error instead of distance, prefetching from pose prediction, or octree-based tiles instead of a fixed grid.
Evaluate on **3 camera paths** and **3 budgets**, with MB and PSNR.

| Criterion | Points |
|---|---|
| Clear hypothesis and implementation | 3 |
| Fair evaluation (same paths and budgets for every method) | 4 |
| Analysis: when does your method win or lose? | 3 |

---

## Capstone — Build a 6DoF streaming experience

Work in teams of 2–3. Choose **one** track:

**A. Static scene streaming.** Stream your own scanned scene (from P1, or a public LiDAR tile) with the Day 15 system. Add at least two improvements, such as better compression, pose-predictive prefetching, semantic priority, cache eviction, or a progressive LoD format.

**B. Volumetric video.** Extend the Day 16 player: per-frame octree compression, an improved ABR algorithm (hybrid, MPC-like), a viewport-dependent quality for the person (front vs back), or multiple people.

**C. Research replication.** Re-implement a simplified idea from a paper on point cloud or Gaussian-splat streaming and evaluate it with our tools.

**Deliverables**
1. A demo that runs from a README in under 5 minutes.
2. A 4-page report: problem, design (with a diagram), experiments (at least 3 network conditions), QoE results, and limitations.
3. A 5-minute talk with a live demo in the last class.

| Criterion | Points |
|---|---|
| Technical quality and originality | 10 |
| Experimental method (baselines, conditions, metrics: bytes, PSNR, stalls, QoE) | 8 |
| Demo works and is understandable | 6 |
| Report and talk clarity | 6 |
