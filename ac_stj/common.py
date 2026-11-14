"""Shared helpers: coordinate frames, AI-line files, INI files."""
import configparser
import io
import math
import re
import struct
from pathlib import Path

import numpy as np

# ── frames ───────────────────────────────────────────────────────────────────
# AC: x, y up, z (left-handed DirectX-style, +z is "south" on the map).  STJ: right-handed, Z up.
# (x, y, z)_AC -> (x, -z, y)_STJ is a pure rotation, so tracks keep their real-world shape
# (checked against ui_track.json "run": clockwise/anticlockwise on all installed tracks).


def ac_to_stj_xyz(x, y, z):
    return (float(x), float(-z), float(y))


def stj_to_ac_xyz(x, y, z):
    return (float(x), float(z), float(-y))


def ac_dir_to_heading(fx, fz):
    """AC forward vector in the ground plane -> STJ heading (rad, CCW from +X)."""
    return math.atan2(-fz, fx)


def heading_to_ac_dir(h):
    return math.cos(h), -math.sin(h)


# ── fast_lane.ai / pit_lane.ai / ideal_line.ai ───────────────────────────────
EXTRA_FIELDS = ("speed", "gas", "brake", "obsolete_lat_g", "radius", "side_left", "side_right", "camber",
                "direction", "normal_x", "normal_y", "normal_z", "length", "forward_x", "forward_y", "forward_z",
                "tag", "grade")


class AiLine:
    """An AC spline: points (n,3) in AC coordinates, dist (n,), and the 18 per-point extras (n,18)."""

    def __init__(self, points, dist=None, extras=None, version=7):
        self.points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        n = len(self.points)
        if dist is None:
            seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
            dist = np.concatenate([[0.0], np.cumsum(seg)])
        self.dist = np.asarray(dist, dtype=np.float64)
        self.extras = None if extras is None else np.asarray(extras, dtype=np.float64).reshape(n, 18)
        self.version = version

    def __len__(self):
        return len(self.points)

    def extra(self, name):
        return None if self.extras is None else self.extras[:, EXTRA_FIELDS.index(name)]

    @classmethod
    def read(cls, path):
        d = Path(path).read_bytes()
        version, n, _, _ = struct.unpack_from("<4i", d, 0)
        rec = np.frombuffer(d, dtype=np.dtype([("p", "<f4", 3), ("d", "<f4"), ("i", "<i4")]), count=n, offset=16)
        extras = None
        off = 16 + n * 20
        if len(d) >= off + 4:
            n2 = struct.unpack_from("<i", d, off)[0]
            if n2 == n and len(d) >= off + 4 + n * 72:
                extras = np.frombuffer(d, dtype="<f4", count=n * 18, offset=off + 4).reshape(n, 18)
        return cls(rec["p"], rec["d"], extras, version)

    def write(self, path):
        n = len(self)
        out = io.BytesIO()
        out.write(struct.pack("<4i", 7, n, 0, 0))
        rec = np.zeros(n, dtype=np.dtype([("p", "<f4", 3), ("d", "<f4"), ("i", "<i4")]))
        rec["p"], rec["d"], rec["i"] = self.points, self.dist, np.arange(n)
        out.write(rec.tobytes())
        out.write(struct.pack("<i", n))
        ex = self.extras if self.extras is not None else default_extras(self.points, self.dist)
        out.write(np.asarray(ex, dtype="<f4").tobytes())
        Path(path).write_bytes(out.getvalue())


def default_extras(points, dist, half_width=6.0, speed=40.0):
    """Plausible per-point AI data for a line that has none (speed from curvature, symmetric sides)."""
    n = len(points)
    ex = np.zeros((n, 18))
    fwd = np.roll(points, -1, axis=0) - points
    ln = np.linalg.norm(fwd, axis=1)
    ln[ln == 0] = 1.0
    fwd = fwd / ln[:, None]
    # curvature radius from the heading change in the ground plane (x, z)
    ang = np.arctan2(fwd[:, 0], fwd[:, 2])
    dang = np.abs(np.angle(np.exp(1j * (np.roll(ang, -1) - ang))))
    radius = np.where(dang > 1e-6, ln / np.maximum(dang, 1e-6), 10000.0)
    radius = np.clip(radius, 5.0, 10000.0)
    v = np.clip(np.sqrt(1.2 * 9.81 * radius), 8.0, 90.0)
    # forward/backward pass: 1 g braking, 0.6 g acceleration
    for _ in range(2):
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2 * 9.81 * ln[i]))
        for i in range(n):
            j = (i - 1) % n
            v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2 * 6.0 * ln[j]))
    ex[:, 0] = v
    ex[:, 1] = 1.0
    ex[:, 4] = radius
    ex[:, 5] = half_width
    ex[:, 6] = half_width
    ex[:, 8] = -1.0
    ex[:, 10] = 1.0
    ex[:, 12] = ln
    ex[:, 13:16] = fwd
    return ex


# ── INI ──────────────────────────────────────────────────────────────────────
def read_ini(path):
    """AC ini: case-preserving keys, ';' and '//' comments, duplicate sections tolerated."""
    cp = configparser.RawConfigParser(strict=False, interpolation=None, inline_comment_prefixes=(";", "//"),
                                      comment_prefixes=(";", "#", "//"), delimiters=("=",))
    cp.optionxform = str
    text = Path(path).read_text(encoding="utf-8", errors="replace").lstrip("\ufeff")
    cp.read_string(text)
    return cp


def ini_sections(cp, prefix):
    """Sections named PREFIX_<n> in numeric order."""
    rx = re.compile(r"^" + re.escape(prefix) + r"_(\d+)$", re.I)
    found = [(int(m.group(1)), s) for s in cp.sections() for m in [rx.match(s)] if m]
    return [cp[s] for _, s in sorted(found)]


def fnum(v, default=0.0):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return default


def fvec(v, n=3):
    parts = [fnum(p) for p in str(v).replace(";", ",").split(",")]
    return (parts + [0.0] * n)[:n]


# ── geometry on a closed/open polyline in the ground plane (STJ x, y) ────────
def project_polyline(xy, chain, p, closed=False, length=None):
    """Nearest point on the polyline: return (s, t, index). t > 0 is left of the travel direction."""
    xy = np.asarray(xy)
    a = xy
    b = np.roll(xy, -1, axis=0) if closed else xy[1:]
    if not closed:
        a = xy[:-1]
    ab = b - a
    ab2 = np.maximum((ab ** 2).sum(1), 1e-12)
    u = np.clip(((np.asarray(p) - a) * ab).sum(1) / ab2, 0.0, 1.0)
    q = a + ab * u[:, None]
    d2 = ((q - np.asarray(p)) ** 2).sum(1)
    i = int(np.argmin(d2))
    seg_len = chain[i + 1] - chain[i] if i + 1 < len(chain) else (length - chain[i] if length else 0.0)
    s = chain[i] + u[i] * seg_len
    cross = ab[i][0] * (np.asarray(p)[1] - a[i][1]) - ab[i][1] * (np.asarray(p)[0] - a[i][0])
    t = math.copysign(math.sqrt(d2[i]), cross)
    return float(s), float(t), i
