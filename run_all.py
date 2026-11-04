"""Professor's helper: run every session script headless and report pass/fail.
Usage:  python run_all.py            (takes ~3-5 minutes)"""
import os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "day01_why_3d/01_dimensions.py", "day01_why_3d/02_parallax_3dof_vs_6dof.py",
    "day02_coordinates_transforms/transforms.py", "day03_first_3d_program/first_program.py",
    "day04_point_clouds_formats/formats_and_sizes.py", "day05_representations/representations.py",
    "day06_capture_camera_model/pinhole_and_rgbd.py", "day07_spatial_structures_lod/kdtree_octree_lod.py",
    "day08_cleaning_registration/clean_and_register.py", "day09_scene_understanding/segment_scene.py",
    "day10_surfaces_neural/mesh_lod_and_splats.py", "day11_rendering_6dof_viewing/frustum_culling.py",
    "day12_streaming_fundamentals/bitrate_buffer_latency.py", "day13_compression/compress_pointcloud.py",
    "day13_compression/vpcc_projection_toy.py", "day14_viewport_adaptive/viewport_adaptive.py",
    "day16_volumetric_video_qoe/abr_volumetric.py", "day16_volumetric_video_qoe/pose_prediction.py",
]
env = dict(os.environ, COURSE_NO_VIZ="1", MPLBACKEND="Agg")
fails = 0
for s in SCRIPTS:
    p = ROOT / s
    t0 = time.time()
    r = subprocess.run([sys.executable, p.name], cwd=p.parent, env=env, capture_output=True, text=True)
    status = "PASS" if r.returncode == 0 else "FAIL"
    fails += r.returncode != 0
    print(f"[{status}] {s:60s} {time.time() - t0:5.1f} s")
    if r.returncode:
        print(r.stderr[-1500:])
print(f"\n{len(SCRIPTS) - fails}/{len(SCRIPTS)} scripts passed.")
