"""STJ document  ->  Assetto Corsa track folder (data/, ai/, ui/, models ini, generated kn5 road mesh)."""
import json
import math
import re
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

from . import __version__
from .common import (AiLine, default_extras, fnum, heading_to_ac_dir, project_polyline, read_ini, stj_to_ac_xyz)
from .kn5_write import Kn5Builder, node_matrix

TOOL = "ac_stj.ac_export"
ISO = {"IT": "Italy", "DE": "Germany", "GB": "United Kingdom", "US": "USA", "FR": "France", "ES": "Spain",
       "BE": "Belgium", "AT": "Austria", "JP": "Japan", "NL": "Netherlands", "BR": "Brazil", "CA": "Canada",
       "AU": "Australia", "SE": "Sweden", "MC": "Monaco", "PT": "Portugal", "HU": "Hungary", "CH": "Switzerland"}
# STJ class -> (AC KEY, FRICTION, IS_VALID_TRACK, dirt additive, wav pitch/ff default)
CLASS_DEFAULT = {"asphalt": ("ROAD", 0.97, 1), "concrete": ("CONCRETE", 0.91, 1), "kerb": ("KERB", 0.93, 1),
                 "painted_line": ("ROAD", 0.9, 1), "cobblestone": ("ROAD", 0.85, 1), "grass": ("GRASS", 0.6, 0),
                 "artificial_grass": ("GRASS", 0.7, 0), "dirt": ("DIRT", 0.75, 0), "gravel": ("GRAVEL", 0.55, 0),
                 "sand": ("SAND", 0.5, 0), "mud": ("MUD", 0.45, 0), "snow": ("SNOW", 0.35, 0), "ice": ("ICE", 0.15, 0),
                 "rock": ("WALL", 0.8, 0), "water": ("WATER", 0.4, 0), "metal": ("ROAD", 0.8, 1), "wood": ("ROAD", 0.7, 1),
                 "wall": ("WALL", 0.8, 0), "other": ("GRASS", 0.6, 0)}
DIFFUSE = {"asphalt": 0.12, "concrete": 0.35, "kerb": 0.6, "grass": 0.2, "gravel": 0.4, "sand": 0.6, "dirt": 0.3}


class _Report:
    KINDS = ("mapped", "approximated", "baked", "dropped", "assumed", "unsupported")

    def __init__(self):
        self.entries = []

    def add(self, kind, path, message):
        self.entries.append({"kind": kind, "path": path, "message": message})

    def counts(self):
        return {k: sum(1 for e in self.entries if e["kind"] == k) for k in self.KINDS}

    def to_dict(self):
        return {"tool": TOOL, "tool_version": __version__, "entries": self.entries}


# ── reference line sampling ──────────────────────────────────────────────────
class Line:
    """The STJ reference polyline with s (chainage), widths and bank, resampled for export."""

    def __init__(self, doc):
        ref = doc.get("reference", {})
        cl = ref.get("centerline")
        if not cl:
            raise ValueError("reference.centerline is required (an OpenDRIVE-only reference is not supported yet)")
        self.closed = bool(cl.get("closed"))
        self.pts = np.array(cl["points"], dtype=float)
        if self.pts.shape[1] == 2:
            self.pts = np.hstack([self.pts, np.zeros((len(self.pts), 1))])
        seg = np.linalg.norm(np.diff(self.pts[:, :2], axis=0), axis=1)
        if "chainage" in cl and len(cl["chainage"]) == len(self.pts):
            self.chain = np.array(cl["chainage"], dtype=float)
        else:
            self.chain = np.concatenate([[0.0], np.cumsum(seg)])
        close = float(np.linalg.norm(self.pts[0, :2] - self.pts[-1, :2])) if self.closed else 0.0
        self.length = float(self.chain[-1] + close)
        if "width_profile" in cl:
            wp = np.array(cl["width_profile"], dtype=float)
            self._ws, self._wl, self._wr = wp[:, 0], wp[:, 1], wp[:, 2]
        else:
            w = cl.get("width", {"left": 6.0, "right": 6.0})
            self._ws, self._wl, self._wr = np.array([0.0]), np.array([w["left"]]), np.array([w["right"]])
        bank = (cl.get("cross_section") or {}).get("bank")
        self.bank = np.array(bank, dtype=float) if bank else None

    def _ext(self):
        """Closed lines get the first point appended so interpolation wraps."""
        if self.closed:
            return np.vstack([self.pts, self.pts[:1]]), np.concatenate([self.chain, [self.length]])
        return self.pts, self.chain

    def at(self, s):
        """(x, y, z, heading, left, right, bank) at chainage(s)."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        if self.closed:
            s = s % self.length
        else:
            s = np.clip(s, 0, self.length)
        p, c = self._ext()
        xyz = np.stack([np.interp(s, c, p[:, k]) for k in range(3)], axis=1)
        eps = 0.5
        s2 = s + eps if not self.closed else (s + eps) % self.length
        s1 = s - eps if not self.closed else (s - eps) % self.length
        a = np.stack([np.interp(np.clip(s1, 0, c[-1]), c, p[:, k]) for k in range(2)], axis=1)
        b = np.stack([np.interp(np.clip(s2, 0, c[-1]), c, p[:, k]) for k in range(2)], axis=1)
        d = b - a
        hdg = np.arctan2(d[:, 1], d[:, 0])
        left = np.interp(s, self._ws, self._wl)
        right = np.interp(s, self._ws, self._wr)
        bank = np.zeros(len(s)) if self.bank is None else np.interp(s, self.bank[:, 0], self.bank[:, 1])
        return xyz, hdg, left, right, bank

    def sample(self, step):
        n = max(int(math.ceil(self.length / step)), 4)
        s = np.arange(n) * (self.length / n)
        if not self.closed:
            s = np.linspace(0, self.length, n + 1)
        return s

    def world(self, s, t):
        """Point at chainage s, offset t to the left."""
        xyz, hdg, *_ = self.at(s)
        x, y, z = xyz[0]
        h = hdg[0]
        return x - t * math.sin(h), y + t * math.cos(h), z, h


def resolve_pose(pose, line):
    """STJ pose -> (x, y, z, heading) in STJ coordinates."""
    if "position" in pose:
        pos = pose["position"]
        return pos[0], pos[1], pos[2], pose["heading"]
    x, y, z, h = line.world(pose["s"], pose["t"])
    return x, y, z, h + pose.get("heading_offset", 0.0)


# ── small writers ────────────────────────────────────────────────────────────
def write_png(path, rgba):
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def draw_map(path, ac_pts, ini_path, pit_pts=None):
    """map.png (white line on transparent, north up as AC draws it: x right, z down) and data/map.ini."""
    margin, scale = 20, 1.0
    xs, zs = ac_pts[:, 0], ac_pts[:, 2]
    w = int(math.ceil(xs.max() - xs.min())) + 2 * margin
    h = int(math.ceil(zs.max() - zs.min())) + 2 * margin
    k = max(1.0, max(w, h) / 1400.0)
    scale = k
    w, h = int(w / k), int(h / k)
    xo, zo = -xs.min() + margin, -zs.min() + margin
    img = np.zeros((h, w, 4), dtype=np.uint8)

    def plot(pts, rad, colour):
        pts = np.vstack([pts, pts[:1]]) if colour is None else pts
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(int(max(abs(b[0] - a[0]), abs(b[2] - a[2])) / k), 1)
            for i in range(n + 1):
                f = i / n
                px = int(((a[0] + (b[0] - a[0]) * f) + xo) / k)
                pz = int(((a[2] + (b[2] - a[2]) * f) + zo) / k)
                img[max(pz - rad, 0):pz + rad + 1, max(px - rad, 0):px + rad + 1] = (255, 255, 255, 255)

    plot(ac_pts, 2, 1 if False else 1)
    write_png(path, img)
    Path(ini_path).write_text(f"[PARAMETERS]\nWIDTH={w}\nHEIGHT={h}\nMARGIN={margin}\nSCALE_FACTOR={scale:g}\n"
                              f"X_OFFSET={xo:.3f}\nZ_OFFSET={zo:.3f}\nDRAWING_SIZE=10\n", encoding="utf-8")


def surf_ini(lib, raw_ini, report):
    """surfaces.ini from the surface library. Sections of a source surfaces.ini with the same KEY are reused."""
    raw = {}
    if raw_ini:
        cp = read_ini_text(raw_ini)
        for sec in cp.sections():
            raw[cp[sec].get("KEY", "").upper()] = dict(cp[sec])
    out, keys = [], {}
    for sid_, s in lib.items():
        cls = s.get("class", "other")
        key0, fr0, valid0 = CLASS_DEFAULT.get(cls, CLASS_DEFAULT["other"])
        key = (s.get("name") or key0).upper().replace(" ", "_")
        if key in keys.values():
            key = key + "_" + sid_.upper()
        keys[sid_] = key
        d = dict(raw.get(key, {})) or {"KEY": key, "FRICTION": fr0, "DAMPING": 0, "WAV": "", "WAV_PITCH": 0,
                                       "FF_EFFECT": "NULL", "DIRT_ADDITIVE": 0, "BLACK_FLAG_TIME": 0,
                                       "IS_VALID_TRACK": valid0, "SIN_HEIGHT": 0, "SIN_LENGTH": 0, "IS_PITLANE": 0,
                                       "VIBRATION_GAIN": 0, "VIBRATION_LENGTH": 0}
        d["KEY"] = key
        mf = s.get("mf_scaling") or {}
        if "LMUX" in mf or "LMUY" in mf:
            d["FRICTION"] = round((mf.get("LMUX", 1.0) + mf.get("LMUY", mf.get("LMUX", 1.0))) / 2, 4)
        if "track_limits" in s:
            d["IS_VALID_TRACK"] = 1 if s["track_limits"] == "inside" else 0
        if s.get("pit_lane"):
            d["IS_PITLANE"] = 1
        out.append(d)
    text = "\n".join("[SURFACE_%d]\n" % i + "\n".join(f"{k}={v}" for k, v in d.items()) + "\n" for i, d in enumerate(out))
    report.add("approximated", "surfaces.ini", "friction = mean of LMUX/LMUY; wet_response, roughness, deformable, "
                                                "by_tyre_class have no AC equivalent")
    return text, keys


def read_ini_text(text):
    import configparser
    cp = configparser.RawConfigParser(strict=False, interpolation=None, delimiters=("=",))
    cp.optionxform = str
    cp.read_string(text)
    return cp


# ── ribbon mesh ──────────────────────────────────────────────────────────────
def surface_intervals(doc, line, s):
    """[(t0, t1, surface_id)] across the track at chainage s: bands (later wins) over a default margin."""
    surf = doc.get("surfaces", {})
    default = surf.get("default")
    bands = surf.get("bands", [])
    _, _, left, right, _ = line.at(s)
    left, right = float(left[0]), float(right[0])
    cuts = {-right - 25.0, left + 25.0, -right, left}
    act = []
    for b in bands:
        a, e = b["s_start"], b["s_end"]
        inside = (a <= s <= e) if a <= e else (s >= a or s <= e)
        if inside:
            act.append(b)
            cuts.update((b["t_min"], b["t_max"]))
    cuts = sorted(cuts)
    res = []
    for t0, t1 in zip(cuts[:-1], cuts[1:]):
        mid = (t0 + t1) / 2
        sid_ = default
        for b in act:
            if b["t_min"] <= mid <= b["t_max"]:
                sid_ = b["surface"]
        if sid_:
            res.append((t0, t1, sid_))
    return res


def build_meshes(doc, line, keys, visual, physics, step=3.0):
    """Adds a ribbon per surface to the visual and physics kn5 builders."""
    s_all = line.sample(step)
    if line.closed:
        s_all = np.concatenate([s_all, [line.length]])
    lib = doc.get("surfaces", {}).get("library", {})
    groups = {}  # surface -> (verts, normals, uvs, indices)
    for i in range(len(s_all) - 1):
        ivs = surface_intervals(doc, line, (s_all[i] + s_all[i + 1]) / 2)
        xyz0, h0, _, _, b0 = line.at(s_all[i])
        xyz1, h1, _, _, b1 = line.at(s_all[i + 1])
        for t0, t1, sid_ in ivs:
            v, n, uv, idx = groups.setdefault(sid_, ([], [], [], []))
            quad = []
            for xyz, h, b, s in ((xyz0[0], h0[0], b0[0], s_all[i]), (xyz1[0], h1[0], b1[0], s_all[i + 1])):
                for t in (t0, t1):
                    tc, ts = math.cos(b), math.sin(b)
                    quad.append((xyz[0] - t * tc * math.sin(h), xyz[1] + t * tc * math.cos(h), xyz[2] + t * ts, t, s))
            base = len(v)
            for (x, y, z, t, s) in quad:
                v.append(stj_to_ac_xyz(x, y, z))
                n.append((0.0, 1.0, 0.0))
                uv.append((t / 10.0, s / 10.0))
            # quad order: (s0,t0) (s0,t1) (s1,t0) (s1,t1)
            for tri in ((0, 2, 1), (1, 2, 3)):
                a, b_, c = (np.array(v[base + k]) for k in tri)
                if np.cross(b_ - a, c - a)[1] < 0:
                    tri = (tri[0], tri[2], tri[1])
                idx.extend(base + k for k in tri)
    for sid_, (v, n, uv, idx) in groups.items():
        cls = lib.get(sid_, {}).get("class", "other")
        key = keys.get(sid_, sid_.upper())
        mi_v = visual.material(sid_, DIFFUSE.get(cls, 0.3))
        mi_p = physics.material(sid_, 0.3)
        v, n, uv, idx = np.array(v), np.array(n), np.array(uv), np.array(idx)
        limit = 60000
        part = 0
        # split into chunks of whole quads (4 vertices each)
        for a in range(0, len(v), limit):
            sel = slice(a, a + limit)
            mask = (idx >= a) & (idx < a + limit)
            tri = idx[mask].reshape(-1, 3)
            tri = tri[(tri >= a).all(1) & (tri < a + limit).all(1)].reshape(-1) - a
            if not len(tri):
                continue
            visual.add_mesh(f"{key}_{part:03d}", v[sel], n[sel], uv[sel], tri, mi_v)
            physics.add_mesh(f"01{key}{part:03d}", v[sel], n[sel], uv[sel], tri, mi_p, visible=False)
            part += 1
    return len(groups)


# ── main entry ───────────────────────────────────────────────────────────────
def stj_to_ac(doc, out_dir, name=None, with_mesh=True, base_dir=None):
    report = _Report()
    line = Line(doc)
    ac = doc.get("x_assetto_corsa", {})
    raw = ac.get("files", {})
    raw_data = {Path(k).name: v for k, v in raw.items() if k.startswith("data/")}
    layout = ac.get("layout") or doc.get("track_info", {}).get("layout") or None
    ti = doc.get("track_info", {})
    name = name or ac.get("track") or re.sub(r"[^a-z0-9_]+", "_", (ti.get("name") or "stj_track").lower()).strip("_")
    root = Path(out_dir) / name
    lay_dir = root / layout if layout else root
    for d in (lay_dir / "data", lay_dir / "ai", root / "ui" / layout if layout else root / "ui"):
        d.mkdir(parents=True, exist_ok=True)
    ui_dir = root / "ui" / layout if layout else root / "ui"
    L = line.length

    # AI lines -----------------------------------------------------------------
    lines = doc.get("race", {}).get("racing_lines", [])
    race_line = next((l for l in lines if l.get("purpose", "race") in ("race", "qualifying")), None)
    step = 1.6
    if race_line and len(race_line["points"]) > 8:
        rp = np.array(race_line["points"], dtype=float)
        if rp.shape[1] == 2:
            rp = np.hstack([rp, np.zeros((len(rp), 1))])
        rseg = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(rp[:, :2], axis=0), axis=1))])
        rclosed = race_line.get("closed", line.closed)
        total = rseg[-1] + (np.linalg.norm(rp[0, :2] - rp[-1, :2]) if rclosed else 0.0)
        n = max(int(total / step), 8)
        sg = np.arange(n) * (total / n) if rclosed else np.linspace(0, total, n)
        rpe = np.vstack([rp, rp[:1]]) if rclosed else rp
        rse = np.concatenate([rseg, [total]]) if rclosed else rseg
        pts = np.stack([np.interp(sg, rse, rpe[:, k]) for k in range(3)], axis=1)
        speed = None
        if race_line.get("speed") and len(race_line["speed"]) == len(rp):
            sp = np.array(race_line["speed"] + ([race_line["speed"][0]] if rclosed else []), dtype=float)
            speed = np.interp(sg, rse, sp)
        report.add("mapped", "race.racing_lines[race]", f"{n} points resampled at ~{step} m")
    else:
        s = line.sample(step)
        xyz, *_ = line.at(s)
        pts, speed = xyz, None
        report.add("assumed", "race.racing_lines", "no race line: AI follows the reference centerline")
    # lateral position of the line w.r.t. the reference -> distance to each edge
    ref2 = line.pts[:, :2]
    sl, sr = np.empty(len(pts)), np.empty(len(pts))
    for i, p in enumerate(pts):
        s_, t_, _ = project_polyline(ref2, line.chain, p[:2], line.closed, L)
        _, _, lft, rgt, _ = line.at(s_)
        sl[i], sr[i] = max(lft[0] - t_, 0.5), max(rgt[0] + t_, 0.5)
    ac_pts = np.array([stj_to_ac_xyz(*p) for p in pts])
    ai = AiLine(ac_pts)
    ex = default_extras(ai.points, ai.dist)
    ex[:, 5], ex[:, 6] = sl, sr
    if speed is not None:
        ex[:, 0] = speed
    ai.extras = ex
    ai.write(lay_dir / "ai" / "fast_lane.ai")
    ai.write(lay_dir / "data" / "ideal_line.ai")
    report.add("approximated", "ai/fast_lane.ai", "speed/radius/side data derived; gas/brake/lateral-g are defaults")

    pit_line = next((l for l in lines if l.get("purpose") in ("pit", "pit_entry", "pit_exit")), None)
    pit_path = pit_line["points"] if pit_line else (doc.get("timing", {}).get("pit_lane", {}) or {}).get("path")
    if pit_path and len(pit_path) > 2:
        pp = np.array([stj_to_ac_xyz(*p) for p in pit_path])
        pl = AiLine(pp)
        pl.extras = default_extras(pl.points, pl.dist, speed=15.0)
        pl.write(lay_dir / "ai" / "pit_lane.ai")
        report.add("mapped", "ai/pit_lane.ai", "from pit racing line / timing.pit_lane.path")
    else:
        report.add("assumed", "ai/pit_lane.ai", "no pit path in the STJ document: file not written (AI cannot pit)")

    # surfaces -----------------------------------------------------------------
    lib = doc.get("surfaces", {}).get("library", {})
    if not lib:
        lib = {"asphalt": {"class": "asphalt", "name": "ROAD", "track_limits": "inside"},
               "grass": {"class": "grass", "name": "GRASS", "track_limits": "outside"}}
        doc = dict(doc, surfaces={"library": lib, "default": "grass",
                                  "bands": [{"s_start": 0, "s_end": L, "t_min": -6, "t_max": 6, "surface": "asphalt"}]})
        report.add("assumed", "surfaces", "no surfaces in the document: asphalt road over grass")
    text, keys = surf_ini(lib, raw_data.get("surfaces.ini"), report)
    (lay_dir / "data" / "surfaces.ini").write_text(text, encoding="utf-8")

    # sections / DRS -----------------------------------------------------------
    race = doc.get("race", {})
    timing = doc.get("timing", {})

    def frac(s):
        return round((s % L) / L, 6) if L else 0.0

    secs = race.get("sections") or []
    txt = []
    for i, s in enumerate(secs):
        t = (s.get("x_assetto_corsa") or {}).get("text") or s["id"]
        txt.append(f"[SECTION_{i}]\nIN={frac(s['s_start'])}\nOUT={frac(s['s_end'])}\nTEXT={t}\n")
    if txt:
        (lay_dir / "data" / "sections.ini").write_text("\n".join(txt), encoding="utf-8")
        report.add("mapped", "sections.ini", f"{len(txt)} race.sections")
    zones = timing.get("drs_zones") or []
    if zones:
        (lay_dir / "data" / "drs_zones.ini").write_text("\n".join(
            f"[ZONE_{i}]\nDETECTION={frac(z['detection_s'])}\nSTART={frac(z['activation_s'])}\nEND={frac(z['deactivation_s'])}\n"
            for i, z in enumerate(zones)), encoding="utf-8")
        report.add("mapped", "drs_zones.ini", f"{len(zones)} zones")

    # cameras ------------------------------------------------------------------
    CAM_DEFAULT = {"SHADOW_SPLIT0": 30, "SHADOW_SPLIT1": 80, "SHADOW_SPLIT2": 200, "MIN_EXPOSURE": 0.2, "MAX_EXPOSURE": 0.45,
                   "DOF_FACTOR": 3, "DOF_RANGE": 150, "DOF_FOCUS": 20, "DOF_MANUAL": 0, "SPLINE": "", "SPLINE_ROTATION": 0,
                   "FOV_GAMMA": 0.5, "SPLINE_ANIMATION_LENGTH": 10, "IS_FIXED": 0, "UP": "0 ,1 ,0"}
    used_names = set()
    for si, cs in enumerate((doc.get("cameras") or {}).get("sets", [])):
        xs = cs.get("x_assetto_corsa") or {}
        hdr = dict(xs.get("header") or {})
        hdr["VERSION"] = hdr.get("VERSION", "3")
        hdr["CAMERA_COUNT"] = len(cs["cameras"])
        hdr["SET_NAME"] = cs.get("name", hdr.get("SET_NAME", cs["id"]))
        blocks = ["[HEADER]\n" + "\n".join(f"{k}={v}" for k, v in hdr.items()) + "\n"]
        for i, c in enumerate(cs["cameras"]):
            pos = np.array(stj_to_ac_xyz(*c["position"]))
            look = np.array(stj_to_ac_xyz(*c["look_at"])) if "look_at" in c else pos + np.array([0, 0, 1.0])
            fw = look - pos
            fw = fw / (np.linalg.norm(fw) or 1.0)
            fov = c.get("fov", [math.radians(40)] * 2)
            sec = {"NAME": i + 1}
            sec.update({k: v for k, v in CAM_DEFAULT.items()})
            sec.update(c.get("x_assetto_corsa") or {})
            sec.update({"POSITION": f"{pos[0]:.6g} ,{pos[1]:.6g} ,{pos[2]:.6g}",
                        "FORWARD": f"{fw[0]:.6g} ,{fw[1]:.6g} ,{fw[2]:.6g}",
                        "MIN_FOV": f"{math.degrees(fov[0]):.6g}", "MAX_FOV": f"{math.degrees(fov[1]):.6g}",
                        "NEAR_PLANE": c.get("near", 1), "FAR_PLANE": c.get("far", 35000)})
            if "active" in c:
                sec["IN_POINT"], sec["OUT_POINT"] = frac(c["active"]["s_start"]), frac(c["active"]["s_end"])
            else:
                sec.setdefault("IN_POINT", -1)
                sec.setdefault("OUT_POINT", -1)
            blocks.append(f"[CAMERA_{i}]\n" + "\n".join(f"{k}={v}" for k, v in sec.items()) + "\n")
        fn = xs.get("file") or ("cameras.ini" if si == 0 else f"cameras_{si}.ini")
        if fn in used_names:
            fn = f"cameras_{si}.ini"
        used_names.add(fn)
        (lay_dir / "data" / fn).write_text("\n".join(blocks), encoding="utf-8")
        report.add("mapped", fn, f"{len(cs['cameras'])} cameras (all AC keys restored from x_assetto_corsa)")

    # verbatim files that need no STJ semantics ---------------------------------
    for fn in ("groove.ini", "startinglights.ini", "overlays.ini", "ai_hints.ini", "camera_facing.ini"):
        if fn in raw_data:
            (lay_dir / "data" / fn).write_text(raw_data[fn], encoding="utf-8")
            report.add("mapped", fn, "restored verbatim from x_assetto_corsa.files")
    from .pacenotes import export_pacenotes
    export_pacenotes(doc, lay_dir / "data", line.length, raw_data, report)
    from .lighting import ext_config_text, lighting_ini
    sun = (doc.get("environment", {}).get("lighting") or {}).get("sun") or {}
    if sun.get("direction"):
        (lay_dir / "data" / "lighting.ini").write_text(lighting_ini(sun, raw_data.get("lighting.ini")), encoding="utf-8")
        report.add("mapped", "lighting.ini", "environment.lighting.sun.direction -> SUN_PITCH/HEADING_ANGLE")
    elif "lighting.ini" not in raw_data:
        (lay_dir / "data" / "lighting.ini").write_text("[LIGHTING]\nSUN_PITCH_ANGLE=35\nSUN_HEADING_ANGLE=300\n", encoding="utf-8")
        report.add("assumed", "lighting.ini", "no sun in the document: default (pitch 35, heading 300)")
    env_doc = doc.get("environment", {})
    if (env_doc.get("lighting") or {}).get("lights") or env_doc.get("geo_reference") or any(k.startswith("extension/") for k in raw):
        raw_ext = next((v for k, v in raw.items() if k.startswith("extension/") and k.endswith("ext_config.ini")), None)
        (root / "extension").mkdir(exist_ok=True)
        (root / "extension" / "ext_config.ini").write_text(ext_config_text(doc, raw_ext), encoding="utf-8")
        nl = len((env_doc.get("lighting") or {}).get("lights", []))
        report.add("mapped", "extension/ext_config.ini", f"{nl} lights + geo position for Custom Shaders Patch "
                                                          "(AC without CSP ignores this file)")

    # grid, pits, timing nodes --------------------------------------------------
    grid = [resolve_pose(g["pose"], line) for g in sorted(race.get("grid", []), key=lambda g: g["position"])]
    if not grid:
        for i in range(24):
            s = (-(8.0 * (i + 1))) % L if line.closed else 8.0 * (i + 1)
            x, y, z, h = line.world(s, 2.5 if i % 2 == 0 else -2.5)
            grid.append((x, y, z, h))
        report.add("assumed", "race.grid", "no grid in the document: 24 slots staggered behind the start line")
    pits = [resolve_pose(p["pose"], line) for p in race.get("pit_boxes", [])]
    if len(pits) < len(grid):
        extra = []
        s0 = L * 0.02 if line.closed else 30.0
        for i in range(len(pits), len(grid)):
            x, y, z, h = line.world((s0 + 8.0 * i) % L, float(line.at(s0)[2][0]) + 10.0)
            extra.append((x, y, z, h))
        pits += extra
        report.add("assumed", "race.pit_boxes", f"{len(extra)} pit boxes generated beside the main straight")
    hot = next((resolve_pose(sp["pose"], line) for sp in race.get("spawns", []) if "hot" in sp.get("id", "")), None)
    if hot is None:
        x, y, z, h = line.world(10.0, 0.0)
        hot = (x, y, z, h)
    sf = (timing.get("start_finish") or timing.get("start") or {}).get("s", 0.0)
    gate_s = [sf] + [sct["s_start"] for sct in timing.get("sectors", [])[1:]] \
        + [sp["s"] for sp in timing.get("split_points", [])]

    def node(label, p):
        x, y, z, h = p
        return label, node_matrix(stj_to_ac_xyz(x, y, z), heading_to_ac_dir(h))

    vis_name = f"{name}" if not layout else f"{name}"
    visual, physics = Kn5Builder(f"{name}.fbx"), Kn5Builder("physics.fbx")
    for i, g in enumerate(grid):
        visual.add_dummy(*node(f"AC_START_{i}", g))
    for i, p in enumerate(pits):
        visual.add_dummy(*node(f"AC_PIT_{i}", p))
    visual.add_dummy(*node("AC_HOTLAP_START_0", hot))
    for i, s in enumerate(gate_s):
        x, y, z, h = line.world(s, 0.0)
        wl = float(line.at(s)[2][0])
        for side, t in (("L", wl), ("R", -float(line.at(s)[3][0]))):
            xx, yy, zz, _ = line.world(s, t)
            visual.add_dummy(*node(f"AC_TIME_{i}_{side}", (xx, yy, zz, h)))
    report.add("mapped", "kn5 nodes", f"{len(grid)} AC_START, {len(pits)} AC_PIT, {len(gate_s)} AC_TIME gate(s)")

    if with_mesh:
        used_usd = False
        if base_dir is not None and doc.get("geometry", {}).get("meshes"):
            from .usd_import import usd_to_kn5
            nv, nc, has_col = usd_to_kn5(doc, base_dir, keys, visual, physics, report)
            used_usd = nv + nc > 0
            if used_usd:
                report.add("mapped", "kn5", f"{nv} visual + {nc} collision meshes from the USD files of geometry.meshes "
                                            "(AC materials restored from customData x_assetto_corsa where present)")
            if used_usd and not has_col:
                ng = build_meshes(doc, line, keys, Kn5Builder("unused"), physics)
                report.add("baked", "1.kn5", f"no collision role in the USD files: {ng} surface ribbons for physics")
        if not used_usd:
            ng = build_meshes(doc, line, keys, visual, physics)
            report.add("baked", "kn5", f"{ng} surface ribbons built from centerline, width, bank and surfaces.bands; "
                                       "no textures, no scenery, no walls, no elevation cross-section, flat colours")
        visual.write(root / f"{name}.kn5")
        physics.write(root / "1.kn5")
        models = f"[MODEL_0]\nFILE=1.kn5\nPOSITION=0,0,0\nROTATION=0,0,0\n\n[MODEL_1]\nFILE={name}.kn5\nPOSITION=0,0,0\nROTATION=0,0,0\n"
        (root / (f"models_{layout}.ini" if layout else "models.ini")).write_text(models, encoding="utf-8")
    else:
        report.add("unsupported", "kn5", "geometry skipped (--no-mesh): bring your own kn5")

    # ui + map -----------------------------------------------------------------
    p2 = np.array([stj_to_ac_xyz(*p) for p in line.pts])
    area = 0.5 * np.sum(line.pts[:, 0] * np.roll(line.pts[:, 1], -1) - np.roll(line.pts[:, 0], -1) * line.pts[:, 1])
    ui = {"name": ti.get("name", name), "description": ti.get("description", ""), "tags": ti.get("tags", []),
          "geotags": ["lat", "lon"], "country": ISO.get(ti.get("country", ""), ti.get("country", "")),
          "city": ti.get("location", ""), "length": str(int(round(L))), "width": str(int(round(
              float(np.median(line._wl + line._wr))))), "pitboxes": str(len(pits)),
          "run": "clockwise" if area < 0 else "anticlockwise"}
    (ui_dir / "ui_track.json").write_text(json.dumps(ui, indent="\t"), encoding="utf-8")
    draw_map(lay_dir / "map.png", p2, lay_dir / "data" / "map.ini")
    write_outline(ui_dir / "outline.png", p2)
    report.add("mapped", "ui/ui_track.json", "name, description, tags, country, location, length, pit boxes, run")
    (root / "conversion.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return root, report


def write_outline(path, ac_pts):
    xs, zs = ac_pts[:, 0], ac_pts[:, 2]
    size = 256
    k = max(xs.max() - xs.min(), zs.max() - zs.min(), 1.0) / (size - 24)
    img = np.zeros((size, size, 4), dtype=np.uint8)
    pts = np.vstack([ac_pts, ac_pts[:1]])
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(int(max(abs(b[0] - a[0]), abs(b[2] - a[2])) / k), 1)
        for i in range(n + 1):
            f = i / n
            px = int((a[0] + (b[0] - a[0]) * f - xs.min()) / k) + 12
            pz = int((a[2] + (b[2] - a[2]) * f - zs.min()) / k) + 12
            img[max(pz - 1, 0):pz + 2, max(px - 1, 0):px + 2] = (255, 255, 255, 255)
    write_png(path, img)
