"""
course3d.py — shared helpers for the "3D & 6DoF Streaming" course.

Every session script imports this file. It gives you:
  * synthetic 3D scenes (so you never need to download data to start),
  * a tiny pinhole-camera renderer written in pure NumPy (so you can SEE
    exactly how 3D points become 2D pixels),
  * simple PLY read/write (no external library needed),
  * a show() helper that opens an Open3D window when possible and always
    saves a PNG preview into the session's ./outputs folder.

Set the environment variable COURSE_NO_VIZ=1 to skip interactive windows
(useful on servers / in CI).  PNG previews are still written.
"""
from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(42)

# ----------------------------------------------------------------------------
# Output folder helpers
# ----------------------------------------------------------------------------

def out_dir(script_file: str) -> Path:
    """Return <session folder>/outputs, creating it if needed."""
    d = Path(script_file).resolve().parent / "outputs"
    d.mkdir(exist_ok=True)
    return d


def viz_enabled() -> bool:
    return os.environ.get("COURSE_NO_VIZ", "0") != "1"


def try_open3d():
    try:
        import open3d as o3d  # noqa: F401
        return o3d
    except Exception:  # pragma: no cover
        return None


# ----------------------------------------------------------------------------
# Synthetic scenes  (all units in metres, Z axis points UP)
# ----------------------------------------------------------------------------

def sample_box(center, size, n, color, rng=RNG, hollow=True):
    """Points on the surface of an axis-aligned box."""
    c = np.asarray(center, float)
    s = np.asarray(size, float)
    if not hollow:
        p = (rng.random((n, 3)) - 0.5) * s + c
        return p, np.tile(color, (n, 1))
    # area-weighted choice of the 6 faces
    areas = np.array([s[1] * s[2], s[1] * s[2], s[0] * s[2], s[0] * s[2], s[0] * s[1], s[0] * s[1]])
    face = rng.choice(6, size=n, p=areas / areas.sum())
    p = (rng.random((n, 3)) - 0.5) * s
    axis = face // 2
    sign = np.where(face % 2 == 0, -0.5, 0.5)
    p[np.arange(n), axis] = sign * s[axis]
    return p + c, np.tile(color, (n, 1)).astype(float)


def sample_sphere(center, radius, n, color=None, rng=RNG):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    p = v * radius + np.asarray(center, float)
    if color is None:  # colour by normal -> pretty and shows the shape
        col = 0.5 + 0.5 * v
    else:
        col = np.tile(color, (n, 1)).astype(float)
    return p, col


def sample_cylinder(base, radius, height, n, color, rng=RNG):
    th = rng.random(n) * 2 * np.pi
    z = rng.random(n) * height
    p = np.c_[radius * np.cos(th), radius * np.sin(th), z] + np.asarray(base, float)
    return p, np.tile(color, (n, 1)).astype(float)


def sample_roof(center, width, depth, height, n, color, rng=RNG):
    """Gable roof: two sloped planes meeting at a ridge parallel to X."""
    u = rng.random(n) - 0.5          # along ridge (x)
    v = rng.random(n)                 # 0 at eave, 1 at ridge
    side = np.where(rng.random(n) < 0.5, -1, 1)
    x = u * width
    y = side * (1 - v) * depth / 2
    z = v * height
    return np.c_[x, y, z] + np.asarray(center, float), np.tile(color, (n, 1)).astype(float)


def sample_plane(xmin, xmax, ymin, ymax, z, n, color, rng=RNG, noise=0.0):
    x = rng.uniform(xmin, xmax, n)
    y = rng.uniform(ymin, ymax, n)
    zz = np.full(n, z) + rng.normal(0, noise, n) if noise else np.full(n, z)
    col = np.tile(color, (n, 1)).astype(float)
    col *= (0.9 + 0.2 * rng.random((n, 1)))  # slight texture
    return np.c_[x, y, zz], np.clip(col, 0, 1)


def make_house(n=20000, origin=(0, 0, 0), rng=RNG):
    """A small house (walls + gable roof + door) — like the book's Ch.1 example.
    Returns points (N,3), colors (N,3) in [0,1], labels (N,) ints."""
    o = np.asarray(origin, float)
    parts, cols, labs = [], [], []
    p, c = sample_box(o + [0, 0, 1.5], [6, 4, 3], int(n * 0.55), [0.85, 0.80, 0.70], rng)
    p = p[p[:, 2] > o[2] + 0.01]  # no floor face
    c = c[: len(p)]
    parts.append(p); cols.append(c); labs.append(np.full(len(p), 1))
    p, c = sample_roof(o + [0, 0, 3], 6.4, 4.4, 1.6, int(n * 0.35), [0.70, 0.20, 0.15], rng)
    parts.append(p); cols.append(c); labs.append(np.full(len(p), 2))
    p, c = sample_box(o + [0, -2.02, 1.0], [1.0, 0.05, 2.0], int(n * 0.05), [0.35, 0.22, 0.12], rng)
    parts.append(p); cols.append(c); labs.append(np.full(len(p), 3))
    p, c = sample_box(o + [-1.8, -2.02, 1.8], [1.0, 0.05, 0.8], int(n * 0.05), [0.45, 0.65, 0.85], rng)
    parts.append(p); cols.append(c); labs.append(np.full(len(p), 4))
    return np.vstack(parts), np.vstack(cols), np.concatenate(labs)


LABEL_NAMES = {0: "ground", 1: "wall", 2: "roof", 3: "door", 4: "window",
               5: "trunk", 6: "foliage", 7: "car", 8: "sphere"}


def make_tree(base, n=3000, rng=RNG):
    p1, c1 = sample_cylinder(base, 0.15, 2.0, n // 4, [0.40, 0.26, 0.13], rng)
    p2, c2 = sample_sphere(np.asarray(base) + [0, 0, 2.8], 1.1, n - n // 4, [0.20, 0.55, 0.20], rng)
    p2 += rng.normal(0, 0.06, p2.shape)
    return np.vstack([p1, p2]), np.vstack([c1, c2]), np.r_[np.full(len(p1), 5), np.full(len(p2), 6)]


def make_scene(n=120_000, rng=RNG, extent=20.0):
    """A small 'street' scene: ground + house + trees + car + a sphere.
    Returns points, colors, labels. ~n points total."""
    P, C, L = [], [], []
    p, c = sample_plane(-extent, extent, -extent, extent, 0.0, int(n * 0.35), [0.45, 0.47, 0.42], rng, noise=0.01)
    P.append(p); C.append(c); L.append(np.zeros(len(p), int))
    p, c, l = make_house(int(n * 0.30), (0, 4, 0), rng)
    P.append(p); C.append(c); L.append(l)
    for tx, ty in [(-8, 2), (8, 6), (-6, -7), (10, -4)]:
        p, c, l = make_tree((tx, ty, 0), int(n * 0.05), rng)
        P.append(p); C.append(c); L.append(l)
    p, c = sample_box((4, -3, 0.75), (4.0, 1.8, 1.5), int(n * 0.08), [0.15, 0.30, 0.75], rng)
    p = p[p[:, 2] > 0.01]
    P.append(p); C.append(c[: len(p)]); L.append(np.full(len(p), 7))
    p, c = sample_sphere((-3, -4, 1.2), 1.2, int(n * 0.07), None, rng)
    P.append(p); C.append(c); L.append(np.full(len(p), 8))
    return np.vstack(P), np.clip(np.vstack(C), 0, 1), np.concatenate(L)


def make_human(t=0.0, n=15000, rng=None):
    """A simple 'volumetric human' made of capsules that waves its arm.
    t = time in seconds. Used for volumetric-video sessions."""
    rng = rng or np.random.default_rng(7)
    P, C = [], []
    def capsule(a, b, r, m, col):
        a, b = np.asarray(a, float), np.asarray(b, float)
        s = rng.random(m)
        axis = b - a
        L = np.linalg.norm(axis)
        d = axis / L
        tmp = np.array([1, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1, 0])
        u = np.cross(d, tmp); u /= np.linalg.norm(u)
        v = np.cross(d, u)
        th = rng.random(m) * 2 * np.pi
        pts = a + np.outer(s, axis) + r * (np.outer(np.cos(th), u) + np.outer(np.sin(th), v))
        P.append(pts); C.append(np.tile(col, (m, 1)))
    skin, shirt, pants = [0.93, 0.76, 0.62], [0.95, 0.55, 0.15], [0.20, 0.30, 0.55]
    sway = 0.05 * np.sin(2 * np.pi * 0.5 * t)
    hip = np.array([sway, 0, 0.95]); neck = np.array([sway, 0, 1.50])
    p, c = sample_sphere(neck + [0, 0, 0.17], 0.12, n // 10, skin, rng); P.append(p); C.append(c)
    capsule(hip, neck, 0.17, n // 4, shirt)
    capsule(hip + [0.1, 0, 0], [0.12, 0.05 * np.sin(2 * np.pi * t), 0.0], 0.07, n // 8, pants)
    capsule(hip + [-0.1, 0, 0], [-0.12, -0.05 * np.sin(2 * np.pi * t), 0.0], 0.07, n // 8, pants)
    sh_l, sh_r = neck + [-0.22, 0, -0.03], neck + [0.22, 0, -0.03]
    capsule(sh_l, sh_l + [-0.08, 0, -0.55], 0.05, n // 10, skin)
    ang = np.pi / 4 + 0.6 * np.sin(2 * np.pi * 1.0 * t)          # waving arm
    elbow = sh_r + [0.30 * np.cos(ang), 0, 0.30 * np.sin(ang)]
    hand = elbow + [0.05, 0, 0.28]
    capsule(sh_r, elbow, 0.05, n // 12, skin)
    capsule(elbow, hand, 0.045, n // 12, skin)
    return np.vstack(P), np.clip(np.vstack(C), 0, 1)


# ----------------------------------------------------------------------------
# Camera maths (OpenCV convention: camera looks along +Z, X right, Y down)
# ----------------------------------------------------------------------------

def intrinsics(width=640, height=480, fov_deg=60.0):
    """Pinhole intrinsic matrix K from image size and horizontal field of view."""
    fx = (width / 2) / np.tan(np.radians(fov_deg) / 2)
    return np.array([[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]], float)


def look_at(eye, target, up=(0, 0, 1)):
    """Return R (3x3) and t (3,) such that X_cam = R @ X_world + t."""
    eye, target, up = map(lambda a: np.asarray(a, float), (eye, target, up))
    z = target - eye; z /= np.linalg.norm(z)            # forward
    x = np.cross(z, up); x /= np.linalg.norm(x)          # right
    y = np.cross(z, x)                                   # down
    R = np.vstack([x, y, z])
    t = -R @ eye
    return R, t


def project(points, K, R, t):
    """3D world points -> pixel coords (u,v) and depth z (camera frame)."""
    Xc = points @ R.T + t
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (Xc[:, :2] / z[:, None]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return uv, z


def render(points, colors, K, R, t, width=640, height=480, point_px=2, bg=(1, 1, 1), out=None):
    """Minimal z-buffer point renderer. Returns (rgb image HxWx3, depth HxW).
    Pass out=(img, depth) from a previous call to draw more points into the same frame."""
    uv, z = project(points, K, R, t)
    ok = z > 0.05
    uv, z, col = uv[ok], z[ok], colors[ok]
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    if out is None:
        img = np.ones((height, width, 3)) * np.asarray(bg, float)
        depth = np.full((height, width), np.inf)
    else:
        img, depth = out
    order = np.argsort(-z)                       # far first, near overwrite
    u, v, z, col = u[order], v[order], z[order], col[order]
    r = point_px // 2
    for du in range(-r, r + 1):
        for dv in range(-r, r + 1):
            uu, vv = u + du, v + dv
            m = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            # depth test: only write where nearer than what's there
            uu, vv, zz, cc = uu[m], vv[m], z[m], col[m]
            nearer = zz < depth[vv, uu]
            img[vv[nearer], uu[nearer]] = cc[nearer]
            depth[vv[nearer], uu[nearer]] = zz[nearer]
    return img, depth


def euler_to_R(yaw, pitch, roll):
    """Z-Y-X (yaw-pitch-roll) Euler angles in radians -> rotation matrix."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def pose_to_view(x, y, z, yaw, pitch, roll):
    """A 6DoF viewer pose (position + yaw/pitch/roll, Z-up world, yaw=0 looks
    along +X, positive pitch = look UP, positive roll = tilt head right) -> (R, t)
    for the renderer above."""
    Rw = euler_to_R(yaw, -pitch, roll)         # body -> world (body: x fwd, y left, z up)
    fwd, left, up = Rw[:, 0], Rw[:, 1], Rw[:, 2]
    R = np.vstack([-left, -up, fwd])           # camera x=right, y=down, z=fwd
    t = -R @ np.array([x, y, z], float)
    return R, t


# ----------------------------------------------------------------------------
# PLY I/O (binary little-endian; float32 xyz + uint8 rgb)
# ----------------------------------------------------------------------------

def write_ply(path, points, colors=None, binary=True):
    path = Path(path)
    n = len(points)
    has_c = colors is not None
    header = ["ply", f"format {'binary_little_endian' if binary else 'ascii'} 1.0",
              f"element vertex {n}", "property float x", "property float y", "property float z"]
    if has_c:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += ["end_header"]
    rgb = (np.clip(colors, 0, 1) * 255).astype(np.uint8) if has_c else None
    if binary:
        dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")] + ([("r", "u1"), ("g", "u1"), ("b", "u1")] if has_c else [])
        arr = np.empty(n, dtype=dt)
        arr["x"], arr["y"], arr["z"] = points[:, 0], points[:, 1], points[:, 2]
        if has_c:
            arr["r"], arr["g"], arr["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        with open(path, "wb") as f:
            f.write(("\n".join(header) + "\n").encode())
            f.write(arr.tobytes())
    else:
        with open(path, "w") as f:
            f.write("\n".join(header) + "\n")
            for i in range(n):
                line = f"{points[i,0]:.4f} {points[i,1]:.4f} {points[i,2]:.4f}"
                if has_c:
                    line += f" {rgb[i,0]} {rgb[i,1]} {rgb[i,2]}"
                f.write(line + "\n")
    return path


def read_ply(path):
    """Read PLY written by write_ply (or Open3D, xyz+rgb float/uchar)."""
    o3d = try_open3d()
    if o3d is not None:
        pc = o3d.io.read_point_cloud(str(path))
        cols = np.asarray(pc.colors) if pc.has_colors() else None
        return np.asarray(pc.points), cols
    raise RuntimeError("Install open3d to read arbitrary PLY files")


# ----------------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------------

def to_o3d(points, colors=None):
    o3d = try_open3d()
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, float)))
    if colors is not None:
        pc.colors = o3d.utility.Vector3dVector(np.clip(np.asarray(colors, float), 0, 1))
    return pc


def preview_png(points, colors, path, eye=None, target=None, title=None, size=(800, 600), point_px=2):
    """Render a quick PNG with the NumPy renderer (works everywhere)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = np.asarray(points)
    c = pts.mean(0) if target is None else np.asarray(target, float)
    if eye is None:
        ext = np.ptp(pts, axis=0).max()
        eye = c + np.array([1.2, -1.6, 0.9]) * ext * 0.8
    W, H = size
    K = intrinsics(W, H, 60)
    R, t = look_at(eye, c)
    cols = colors if colors is not None else np.tile([0.2, 0.3, 0.8], (len(pts), 1))
    img, _ = render(pts, cols, K, R, t, W, H, point_px)
    plt.figure(figsize=(W / 100, H / 100), dpi=100)
    plt.imshow(img); plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout(); plt.savefig(path); plt.close()
    return path


def show(geoms, title="3D view", png_path=None):
    """Open an Open3D window (if available & allowed). `geoms` is a list of
    Open3D geometries or (points, colors) tuples."""
    o3d = try_open3d()
    if png_path is not None:
        # save a PNG of the first point-like geometry for the record
        g0 = geoms[0]
        if isinstance(g0, tuple):
            preview_png(g0[0], g0[1], png_path, title=title)
        elif o3d is not None and isinstance(g0, o3d.geometry.PointCloud):
            preview_png(np.asarray(g0.points), np.asarray(g0.colors) if g0.has_colors() else None, png_path, title=title)
    if not viz_enabled() or o3d is None:
        if o3d is None:
            print("[info] open3d not installed -> no interactive window (PNG saved instead)")
        return
    gl = [to_o3d(*g) if isinstance(g, tuple) else g for g in geoms]
    try:
        o3d.visualization.draw_geometries(gl, window_name=title, width=1100, height=750)
    except Exception as e:  # e.g. no display
        print(f"[info] could not open a window ({e}); see the PNG in outputs/")


def human_bytes(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def banner(text):
    print("\n" + "=" * 72 + f"\n  {text}\n" + "=" * 72)
