# Spatial Media Engineering: From 2D Pixels to 3D Streams (16 sessions)

Hands-on companion code for the 16-session course **"From 3D Basics to 6DoF Streaming"**.
Sessions 1–10 follow *3D Data Science with Python* (Florent Poux, O'Reilly 2025);
sessions 11–16 extend it to rendering, compression and 6DoF / volumetric streaming.

Everything runs **offline**: every script builds its own synthetic 3D scene
(a small street with a house, trees, a car, a sphere, and a waving person for
volumetric video), so nobody needs to download datasets to start. Swap in the
book's datasets or your own scans whenever you like.

## 1. Setup (once, ~10 minutes)

```bash
# Python 3.10 – 3.12 recommended (Open3D wheels are available for these)
python -m venv venv3d
# Windows: venv3d\Scripts\activate      macOS/Linux: source venv3d/bin/activate
pip install -r requirements.txt
python check_setup.py
```

The browser demos (sessions 1, 11, 15, 16) need only a modern browser
(Chrome, Edge or Firefox) and an internet connection to load three.js from a CDN.

## 2. How to run a session

```bash
cd day05_representations
python representations.py
```

* Text results print in the terminal.
* Figures and `.ply` files are written to that session's `outputs/` folder.
* An interactive Open3D 3D window opens where useful (rotate: left-drag,
  pan: Ctrl/Shift + drag, zoom: scroll, close the window to continue).
  Set `COURSE_NO_VIZ=1` to skip windows (e.g. on a server).
* `.ply` files open in MeshLab, CloudCompare, or https://3dviewer.net.

## 3. The 16 sessions

| Day | Topic | Run this | Book |
|----:|-------|----------|------|
| 01 | What is 3D? Why 3D? 3DoF vs 6DoF | `01_dimensions.py`, `02_parallax_3dof_vs_6dof.py`, open `dof_playground.html` | Ch. 1 |
| 02 | Coordinates, transformations, the 6DoF pose | `transforms.py` | Ch. 2 |
| 03 | Your first 3D Python program | `first_program.py` | Ch. 3 |
| 04 | Point clouds, file formats and bytes | `formats_and_sizes.py` | Ch. 3–4 |
| 05 | Representations: points, meshes, voxels, depth maps | `representations.py` | Ch. 4 |
| 06 | Capturing 3D: pinhole camera and RGB-D | `pinhole_and_rgbd.py` | Ch. 1, 3, 10 |
| 07 | k-d trees, octrees and Levels of Detail | `kdtree_octree_lod.py` | Ch. 4, 17 |
| 08 | Cleaning and registration (ICP = finding a 6DoF pose) | `clean_and_register.py` | Ch. 5–6 |
| 09 | Scene understanding: features, RANSAC, DBSCAN | `segment_scene.py` | Ch. 7, 9, 12 |
| 10 | Meshing, mesh LoDs, Gaussian splats | `mesh_lod_and_splats.py` | Ch. 10, 18 |
| 11 | Rendering and the view frustum; a 6DoF viewer | `frustum_culling.py`, then `viewer6dof/` | Ch. 5, 17 |
| 12 | Streaming basics: bitrate, buffer, latency | `bitrate_buffer_latency.py` | — |
| 13 | Point cloud compression (G-PCC and V-PCC ideas) | `compress_pointcloud.py`, `vpcc_projection_toy.py` | Ch. 4, 17 |
| 14 | Viewport-adaptive 6DoF streaming | `viewport_adaptive.py` | — |
| 15 | A live 6DoF streaming system | `server.py`, then open http://localhost:8000 | — |
| 16 | Volumetric video, ABR, QoE, pose prediction | `abr_volumetric.py`, `pose_prediction.py`, `vv_server.py` | Ch. 18 |

Each `dayXX_*/README.md` says what to look at and gives **"Try this"** mini-experiments.
The graded work is in `ASSIGNMENTS.md`.

## 4. Browser demos

```bash
# Day 01 & 11 (static pages)
cd day11_rendering_6dof_viewing
python frustum_culling.py          # creates viewer6dof/scene.ply
cd viewer6dof && python -m http.server 8000
# open http://localhost:8000

# Day 15 (live tiled 6DoF streaming)
cd day15_live_6dof_streaming && python server.py
# open http://localhost:8000

# Day 16 (volumetric video with ABR)
cd day16_volumetric_video_qoe && python vv_server.py
# open http://localhost:8001
```

Controls everywhere: **drag** = look, **W A S D** = move, **R / F** = up/down,
**Q / E** = roll, **Shift** = faster.

## 5. Conventions you will meet

* Python scripts use a **Z-up** world (like LiDAR/GIS and Open3D examples).
  three.js in the browser uses **Y-up**; the code converts `(x, y, z) -> (x, z, -y)`.
  Mixing up conventions is the #1 bug in 3D — Session 2 makes this explicit.
* Units are **metres**. Angles in code are **radians**; printouts show degrees.
* `common/course3d.py` holds shared helpers (scenes, a NumPy renderer, PLY I/O).
  It is short and commented — read it!

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `pip install open3d` fails | Use Python 3.10–3.12 (`py -3.11 -m venv venv3d` on Windows). |
| No 3D window appears | Normal on servers/WSL without display; open the PNG in `outputs/`. |
| Browser page is blank | Serve it with `python -m http.server` (don't double-click the file) and check internet access for the three.js CDN. |
| Port already in use | `python server.py --port 8010` |

## 7. Slide decks (with speaker notes for the professor)

| Days | Deck |
|---|---|
| 01 | https://claude.ai/artifact/W752DcRd7T1rQBxSYHwe7R |
| 02 | https://claude.ai/artifact/C6tKMt8uL2Y5h6s1uYV1pB |
| 03 | https://claude.ai/artifact/XgANYiuE3CbmTiqXCuT1cA |
| 04 | https://claude.ai/artifact/2bLkwLH1cCb9vhq8w2irFC |
| 05 | https://claude.ai/artifact/KrKtYboxAs4YyJ97tueDG8 |
| 06 | https://claude.ai/artifact/8dkP2idokvzxTgsQ9t8ZDR |
| 07 | https://claude.ai/artifact/SiZPYi42dTu5Ag8q3LT9Tt |
| 08 | https://claude.ai/artifact/W5avRjvULpsNQeEYaqRTyc |
| 09–16 (one deck, one section per day) | https://claude.ai/artifact/4cYga5v1dd6evXhiuJYD96 |

Assignments (editable doc): https://claude.ai/artifact/95GD5U3FjJcgb5EsisKiRT — the same content is in `ASSIGNMENTS.md`.
