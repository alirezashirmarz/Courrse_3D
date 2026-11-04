"""
Day 03 — Your first 3D Python program  (book Ch.3 "Creating a 3D Python Program")
------------------------------------------------------------------------------
Run:  python first_program.py

The 4 classic steps from the book:
  1. Import 3D data           (an ASCII file: X;Y;Z;R;G;B;class)
  2. Extract attributes       (coordinates, colours, labels)
  3. Attribute-based analysis (statistics, filtering by height / class)
  4. Visualise and export     (Open3D window + .ply files)

The script first CREATES the data file (data/street_scene.txt) so it works
offline. Replace it with any of the book's datasets later — same code!
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import course3d as c3d

HERE = Path(__file__).resolve().parent
OUT = c3d.out_dir(__file__)
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)
c3d.banner("Day 03 · First 3D program")

# 0) make a data file exactly like the book's ASCII examples -----------------------
f = DATA / "street_scene.txt"
if not f.exists():
    P, C, L = c3d.make_scene(60_000)
    arr = np.c_[P, (C * 255).round(), L]
    np.savetxt(f, arr, delimiter=";", fmt=["%.3f"] * 3 + ["%d"] * 4, header="X;Y;Z;R;G;B;class", comments="")
    print("created", f)

# 1) IMPORT ---------------------------------------------------------------------
pcd_np = np.loadtxt(f, delimiter=";", skiprows=1)
print("Loaded array shape:", pcd_np.shape, "(rows = points, columns = attributes)")
print("First 3 points:\n", pcd_np[:3])

# 2) EXTRACT ATTRIBUTES -----------------------------------------------------------
xyz = pcd_np[:, 0:3]
rgb = pcd_np[:, 3:6] / 255.0
labels = pcd_np[:, 6].astype(int)

# 3) ANALYSIS ---------------------------------------------------------------------
print("\nBounding box min:", xyz.min(0).round(2), " max:", xyz.max(0).round(2))
print("Extent (m):", np.ptp(xyz, axis=0).round(2))
print("Mean height: %.2f m, highest point: %.2f m" % (xyz[:, 2].mean(), xyz[:, 2].max()))
print("\nPoints per class:")
for k, n in zip(*np.unique(labels, return_counts=True)):
    print(f"   {k}  {c3d.LABEL_NAMES.get(k, '?'):8s} {n:7,d} points   mean z = {xyz[labels == k, 2].mean():.2f} m")

above = xyz[:, 2] > 0.3                      # everything that is NOT ground
print(f"\nHeight filter z > 0.3 m keeps {above.sum():,} of {len(xyz):,} points ({above.mean():.0%})")
tall = xyz[:, 2] > 3.0
print(f"Points above 3 m (roofs, tree crowns): {tall.sum():,}")

# 4) VISUALISE + EXPORT -----------------------------------------------------------
c3d.write_ply(OUT / "scene_full.ply", xyz, rgb)
c3d.write_ply(OUT / "scene_no_ground.ply", xyz[above], rgb[above])
# colour by height to "see" an attribute
h = (xyz[:, 2] - xyz[:, 2].min()) / np.ptp(xyz[:, 2])
import matplotlib.cm as cm
height_col = cm.turbo(h)[:, :3]
c3d.write_ply(OUT / "scene_height_coloured.ply", xyz, height_col)
print("\nExported PLY files to", OUT)
c3d.show([(xyz[above], rgb[above])], "Day 03 — scene without ground", OUT / "no_ground.png")
c3d.show([(xyz, height_col)], "Day 03 — coloured by height", OUT / "height.png")
print("Tip: drag the .ply files into MeshLab / CloudCompare / https://3dviewer.net to view them too.")
