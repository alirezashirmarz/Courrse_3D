"""
Day 13 (part 2) — The V-PCC idea: turn 3D into 2D images, then use a video codec
-------------------------------------------------------------------------------
Run:  python vpcc_projection_toy.py

MPEG V-PCC (Video-based Point Cloud Compression) projects the point cloud
onto the sides of a box, producing GEOMETRY images (depth per pixel) and
TEXTURE images (colour per pixel).  Those are just pictures -> any video
codec (HEVC/H.265) can compress them, and every phone has a hardware
video decoder.  This toy version uses 6 projections + PNG/JPEG.
"""
import io, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import cKDTree
import course3d as c3d

OUT = c3d.out_dir(__file__)
c3d.banner("Day 13 · V-PCC projection toy")
P, C = c3d.make_human(0.0, 60_000)
RES = 512                                              # image resolution
EMPTY = 65535                                          # marks 'no point here'
lo = P.min(0); span = np.ptp(P, axis=0).max() * 1.001
q = ((P - lo) / span * (RES - 1)).round().astype(int)   # 8-bit-ish voxel grid

# 6 projection directions: (axis, +/-)
faces = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]
geo_imgs, tex_imgs = [], []
assigned = np.zeros(len(P), bool)
normals_dir = q - q.mean(0)
best_face = np.argmax(np.c_[[s * normals_dir[:, a] for a, s in faces]].T, axis=1)   # crude "patch" assignment
for fi, (axis, sign) in enumerate(faces):
    u_ax, v_ax = [a for a in range(3) if a != axis]
    sel = np.where(best_face == fi)[0]
    depth = np.full((RES, RES), EMPTY, np.uint16)   # 16-bit depth image
    tex = np.zeros((RES, RES, 3), np.uint8)
    d = q[sel, axis] if sign < 0 else (RES - 1 - q[sel, axis])
    order = np.argsort(-d)                               # nearest to the face wins
    s = sel[order]
    depth[q[s, v_ax], q[s, u_ax]] = d[order]
    tex[q[s, v_ax], q[s, u_ax]] = (C[s] * 255).astype(np.uint8)
    geo_imgs.append(depth); tex_imgs.append(tex)

def png_size(a): b = io.BytesIO(); Image.fromarray(a).save(b, "PNG", optimize=True); return len(b.getvalue())
def jpg(a, qual): b = io.BytesIO(); Image.fromarray(a).save(b, "JPEG", quality=qual); return len(b.getvalue()), np.array(Image.open(io.BytesIO(b.getvalue())))

# reconstruct from images
def reconstruct(geo, tex):
    pts, cols = [], []
    for (axis, sign), dimg, timg in zip(faces, geo, tex):
        u_ax, v_ax = [a for a in range(3) if a != axis]
        vv, uu = np.nonzero(dimg != EMPTY)
        d = dimg[vv, uu].astype(int)
        g = np.zeros((len(vv), 3)); g[:, u_ax] = uu; g[:, v_ax] = vv
        g[:, axis] = d if sign < 0 else RES - 1 - d
        pts.append(g); cols.append(timg[vv, uu] / 255)
    return np.vstack(pts) / (RES - 1) * span + lo, np.vstack(cols)

rec, rec_col = reconstruct(geo_imgs, tex_imgs)
geo_bytes = sum(png_size(g) for g in geo_imgs)
tex_sizes = [jpg(t, 80) for t in tex_imgs]
tex_bytes = sum(s for s, _ in tex_sizes)
rec2, rec2_col = reconstruct(geo_imgs, [t for _, t in tex_sizes])
d, _ = cKDTree(P).query(rec)
print(f"Input: {len(P):,} points = {c3d.human_bytes(len(P) * 15)} raw")
print(f"6 geometry images (PNG, lossless): {c3d.human_bytes(geo_bytes)}")
print(f"6 texture images (JPEG q=80):      {c3d.human_bytes(tex_bytes)}")
print(f"Total {c3d.human_bytes(geo_bytes + tex_bytes)} -> {(geo_bytes + tex_bytes) * 8 / len(P):.2f} bits/point"
      f"  (compression x{len(P) * 15 / (geo_bytes + tex_bytes):.0f})")
print(f"  = {(geo_bytes + tex_bytes) * 8 / len(rec):.2f} bits per RECONSTRUCTED point")
print(f"Reconstructed {len(rec):,} points (occluded points within a projection are LOST -> real V-PCC uses 2 layers + patches)")
print(f"Mean geometric error: {d.mean() * 1000:.1f} mm")

fig, ax = plt.subplots(2, 7, figsize=(21, 6.5))
for i in range(6):
    ax[0, i].imshow(np.where(geo_imgs[i] == EMPTY, np.nan, geo_imgs[i]), cmap="gray"); ax[0, i].set_title(f"geometry {['+','-'][faces[i][1] < 0]}{'XYZ'[faces[i][0]]}")
    ax[1, i].imshow(tex_imgs[i]); ax[1, i].set_title("texture")
K = c3d.intrinsics(300, 400, 40); R, t = c3d.look_at((1.8, -3.5, 1.4), (0, 0, 0.9))
ax[0, 6].imshow(c3d.render(P, C, K, R, t, 300, 400, 2)[0]); ax[0, 6].set_title("original")
ax[1, 6].imshow(c3d.render(rec2, rec2_col, K, R, t, 300, 400, 2)[0]); ax[1, 6].set_title("decoded from images")
for a in ax.ravel(): a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "vpcc_toy.png", dpi=90); plt.close()
print("saved", OUT / "vpcc_toy.png")
