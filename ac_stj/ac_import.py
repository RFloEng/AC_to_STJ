"""Assetto Corsa track folder  ->  STJ document (+ conversion report, spec 3.12)."""
import datetime
import json
import math
import re
from pathlib import Path

import numpy as np

from . import __version__
from .common import (AiLine, ac_dir_to_heading, ac_to_stj_xyz, fnum, fvec, ini_sections, project_polyline,
                     read_ini)
from .kn5 import read_kn5, world_forward_xz, world_position

TOOL = "ac_stj.ac_import"

COUNTRY = {"italy": "IT", "germany": "DE", "united kingdom": "GB", "uk": "GB", "england": "GB", "usa": "US",
           "united states": "US", "france": "FR", "spain": "ES", "belgium": "BE", "austria": "AT", "japan": "JP",
           "netherlands": "NL", "brazil": "BR", "canada": "CA", "australia": "AU", "sweden": "SE", "monaco": "MC",
           "portugal": "PT", "hungary": "HU", "hungary ": "HU", "switzerland": "CH", "czech republic": "CZ"}

# AC surface KEY -> STJ surface class.  First regex that matches wins.
CLASS_RULES = [
    (r"kerb|curb|rumble", "kerb"), (r"concrete|cement|pit|pavement|road|asph|tarmac|track|carpet|tar", "asphalt"),
    (r"grass|lawn|green", "grass"), (r"gravel|stone|pebble", "gravel"), (r"sand", "sand"), (r"mud", "mud"),
    (r"snow", "snow"), (r"ice", "ice"), (r"dirt|earth|soil|ground|clay", "dirt"), (r"rock", "rock"),
    (r"wall|barrier|tire|tyre", "wall"), (r"water", "water"), (r"metal|steel", "metal"), (r"wood", "wood"),
    (r"line|paint", "painted_line"), (r"cobble|sett", "cobblestone"),
]
PRIMARY = {"asphalt": "tarmac", "concrete": "tarmac", "kerb": "tarmac", "gravel": "gravel", "dirt": "dirt",
           "snow": "snow", "ice": "ice", "sand": "dirt", "mud": "dirt"}


def sid(text, fallback="x"):
    s = re.sub(r"[^a-z0-9_]+", "_", str(text).lower()).strip("_")
    return s or fallback


def classify(key):
    for rx, cls in CLASS_RULES:
        if re.search(rx, key, re.I):
            return cls
    return "other"


def list_layouts(track_dir):
    """Layout names of an AC track folder; [None] when the track has a single, root-level layout."""
    track_dir = Path(track_dir)
    if (track_dir / "ai" / "fast_lane.ai").exists():
        return [None]
    return sorted(p.name for p in track_dir.iterdir() if p.is_dir() and (p / "ai" / "fast_lane.ai").exists())


def _read_text(path):
    return Path(path).read_text(encoding="utf-8", errors="replace").lstrip("﻿")


def _models_for(track_dir, layout):
    """kn5 files of this layout (models_<layout>.ini, else models.ini, else every kn5 in the folder)."""
    cand = ([track_dir / f"models_{layout}.ini"] if layout else []) + [track_dir / "models.ini"]
    for c in cand:
        if c.exists():
            cp = read_ini(c)
            files = [cp[s].get("FILE") for s in cp.sections() if cp[s].get("FILE")]
            return [track_dir / f for f in files if (track_dir / f).exists()]
    return sorted(track_dir.glob("*.kn5"))


def read_nodes(track_dir, layout, report):
    """Named empties (AC_START_*, AC_PIT_*, AC_TIME_*, AC_HOTLAP_START_*, AC_AB_*) of the layout's kn5 files, in STJ frame."""
    nodes = {}
    for f in _models_for(track_dir, layout):
        try:
            _, root = read_kn5(f)
        except Exception as e:  # unreadable/unknown kn5 variant: keep going with the data files
            report.add("unsupported", f.name, f"kn5 node tree not read: {e}")
            continue
        for n, w in root.walk():
            if n.kind == "dummy" and re.match(r"AC_(START|PIT|TIME|HOTLAP_START|AB_START|AB_FINISH)", n.name):
                p = ac_to_stj_xyz(*world_position(w))
                fx, fz = world_forward_xz(w)
                nodes[n.name] = {"position": p, "heading": ac_dir_to_heading(fx, fz)}
    return nodes


def ac_to_stj(track_dir, layout=None, include_raw=True, meshes=None):
    """Convert one layout. Returns (stj_document, ConversionReport).

    meshes: None, or (assets_dir, uri_base, textures) to also convert the kn5 meshes to USD (needs usd-core)."""
    import sys
    sys.path.insert(0, str(_stj_tools()))
    from stj_core import ConversionReport  # noqa: E402

    track_dir = Path(track_dir)
    base = track_dir / layout if layout else track_dir
    data, ai_dir = base / "data", base / "ai"
    report = ConversionReport(TOOL, str(track_dir), "", __version__)
    raw = {}  # verbatim text of the small data files, kept in x_assetto_corsa for a lossless return trip

    def keep(p):
        if p.exists() and p.stat().st_size < 200_000:
            raw[str(p.relative_to(base)).replace("\\", "/")] = _read_text(p)

    # ── reference line ───────────────────────────────────────────────────────
    fl = ai_dir / "fast_lane.ai"
    if not fl.exists() or fl.stat().st_size < 100:
        raise ValueError(f"{track_dir.name}/{layout or ''}: no usable ai/fast_lane.ai (stub or missing); "
                         "the reference line comes from it")
    fast = AiLine.read(fl)
    pts = np.array([ac_to_stj_xyz(*p) for p in fast.points])
    n = len(pts)
    gap = float(np.linalg.norm(pts[0] - pts[-1]))
    seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    mean_seg = float(seg.mean()) if len(seg) else 1.0
    closed = gap < 3.0 * mean_seg + 1.0
    chain = fast.dist.copy()
    if not (chain[0] == 0 and np.all(np.diff(chain) > 0)):
        chain = np.concatenate([[0.0], np.cumsum(seg)])
        report.add("assumed", "/reference/centerline/chainage", "AI distances not monotonic; recomputed from geometry")
    length = float(chain[-1] + (np.linalg.norm(pts[0, :2] - pts[-1, :2]) if closed else 0.0))
    cl = {"closed": closed, "points": [[round(float(v), 4) for v in p] for p in pts],
          "chainage": [round(float(c), 4) for c in chain]}
    sl, sr = fast.extra("side_left"), fast.extra("side_right")
    if sl is not None and np.all(sl >= 0) and np.all(sr >= 0):
        wp = [[round(float(chain[i]), 3), round(float(sl[i]), 3), round(float(sr[i]), 3)] for i in range(0, n)]
        cl["width_profile"] = wp
        report.add("mapped", "/reference/centerline/width_profile", "AI side_left/side_right (distance to the track edge)")
    else:
        cl["width"] = {"left": 6.0, "right": 6.0}
        report.add("assumed", "/reference/centerline/width", "no AI side data; 6 m half-width")
    report.add("mapped", "/reference/centerline", f"{n} points of ai/fast_lane.ai; z up, AC (x,y,z) -> STJ (x,-z,y)")

    doc = {"_metadata": {"specification": "STJ", "version": "0.3", "coordinate_system": "OpenDRIVE", "units": "SI",
                         "source_format": "Assetto Corsa"},
           "reference": {"centerline": cl, "length": round(length, 3)}}

    def s_of(frac):
        return round((fnum(frac) % 1.0 if fnum(frac) != 1.0 else 1.0) * length, 3)

    def project(p):
        return project_polyline(pts[:, :2], chain, p[:2], closed, length)

    # ── track_info ───────────────────────────────────────────────────────────
    ui_path = _find_ui(track_dir, layout)
    ui = {}
    if ui_path:
        try:
            ui = json.loads(_read_text(ui_path), strict=False)
        except json.JSONDecodeError:
            report.add("unsupported", str(ui_path.name), "ui_track.json is not valid JSON")
        raw[str(ui_path.relative_to(track_dir)).replace("\\", "/")] = _read_text(ui_path)
    ti = {"name": ui.get("name") or track_dir.name, "type": "circuit" if closed else "point_to_point",
          "length": round(length, 3)}
    if layout:
        ti["layout"] = layout
    if ui.get("description"):
        ti["description"] = ui["description"]
    if ui.get("tags"):
        ti["tags"] = sorted(set(str(t) for t in ui["tags"]))
    c = str(ui.get("country", "")).strip().lower()
    if c in COUNTRY:
        ti["country"] = COUNTRY[c]
    elif len(c) == 2 and c.isalpha():
        ti["country"] = c.upper()
    if ui.get("city"):
        ti["location"] = str(ui["city"])
    doc["track_info"] = ti
    if ui.get("run") in ("clockwise", "anticlockwise", "counterclockwise"):
        doc["track_info"]["x_assetto_corsa"] = {"run": ui["run"]}
    report.add("mapped", "/track_info", "ui_track.json name, description, tags, country, city")

    # ── surfaces ─────────────────────────────────────────────────────────────
    lib, keys = {}, {}
    sp = data / "surfaces.ini"
    if sp.exists():
        keep(sp)
        cp = read_ini(sp)
        for sec in ini_sections(cp, "SURFACE"):
            key = sec.get("KEY", "")
            sid_ = sid(key, "surface")
            while sid_ in lib:
                sid_ += "_"
            fr = fnum(sec.get("FRICTION"), 1.0)
            cls = classify(key)
            surf = {"name": key, "class": cls, "mf_scaling": {"LMUX": round(fr, 4), "LMUY": round(fr, 4), "_est": True},
                    "track_limits": "inside" if fnum(sec.get("IS_VALID_TRACK"), 1) else "outside"}
            if fnum(sec.get("IS_PITLANE")):
                surf["pit_lane"] = True
            if fnum(sec.get("DAMPING")) > 0:
                surf["rolling_resistance"] = round(fnum(sec.get("DAMPING")) * 0.001, 5)
                report.add("approximated", f"/surfaces/library/{sid_}/rolling_resistance",
                           "AC DAMPING scaled by 1e-3; AC damping is not a rolling-resistance coefficient")
            sin_h, sin_l = fnum(sec.get("SIN_HEIGHT")), fnum(sec.get("SIN_LENGTH"))
            if sin_h > 0 and sin_l > 0:
                surf["x_assetto_corsa"] = {"sin_height": sin_h, "sin_length": sin_l}
            lib[sid_] = surf
            keys[key] = sid_
        report.add("approximated", "/surfaces/library",
                   "AC FRICTION used as LMUX/LMUY (marked _est); class guessed from the surface KEY; "
                   "WAV, FF_EFFECT, DIRT_ADDITIVE, BLACK_FLAG_TIME, VIBRATION_* only in x_assetto_corsa")
    surfaces = {"library": lib}
    if lib:
        valid = [k for k, v in lib.items() if v["track_limits"] == "inside" and not v.get("pit_lane")
                 and v["class"] in ("asphalt", "concrete")]
        outside = [k for k, v in lib.items() if v["track_limits"] == "outside"]
        road = valid[0] if valid else next(iter(lib))
        surfaces["default"] = outside[0] if outside else road
        if sl is not None:
            surfaces["bands"] = [{"s_start": 0.0, "s_end": round(length, 3),
                                  "t_min": -round(float(np.median(sr)), 2), "t_max": round(float(np.median(sl)), 2),
                                  "surface": road}]
            report.add("assumed", "/surfaces/bands",
                       "AC assigns surfaces by mesh name inside the kn5; without geometry export the track "
                       "width is one band of the first valid road surface")
    doc["surfaces"] = surfaces

    # ── nodes: grid, pits, start line ────────────────────────────────────────
    nodes = read_nodes(track_dir, layout, report)
    race = {}
    starts = sorted(((int(m.group(1)), v) for k, v in nodes.items() for m in [re.fullmatch(r"AC_START_(\d+)", k)] if m))
    if starts:
        race["grid"] = [{"position": i + 1, "pose": {"position": [round(c, 3) for c in v["position"]],
                                                    "heading": round(v["heading"], 5)}} for i, (_, v) in enumerate(starts)]
        report.add("mapped", "/race/grid", f"{len(starts)} AC_START_n nodes (pole = AC_START_0)")
    pits = sorted(((int(m.group(1)), v) for k, v in nodes.items() for m in [re.fullmatch(r"AC_PIT_(\d+)", k)] if m))
    if pits:
        race["pit_boxes"] = [{"id": f"pit_{i}", "pose": {"position": [round(c, 3) for c in v["position"]],
                                                         "heading": round(v["heading"], 5)}} for i, v in pits]
        report.add("mapped", "/race/pit_boxes", f"{len(pits)} AC_PIT_n nodes")
    hot = nodes.get("AC_HOTLAP_START_0")
    if hot:
        race["spawns"] = [{"id": "hotlap_start", "pose": {"position": [round(c, 3) for c in hot["position"]],
                                                          "heading": round(hot["heading"], 5)}}]
    timing = {}
    ab_a, ab_b = nodes.get("AC_AB_START_L"), nodes.get("AC_AB_START_R")
    gates = {}
    for k, v in nodes.items():
        m = re.fullmatch(r"AC_TIME_(\d+)_([LR])", k)
        if m:
            gates.setdefault(int(m.group(1)), {})[m.group(2)] = v["position"]
    gate_s = {}
    for i, lr in gates.items():
        if "L" in lr and "R" in lr:
            mid = np.array([(lr["L"][0] + lr["R"][0]) / 2, (lr["L"][1] + lr["R"][1]) / 2])
            gate_s[i] = project(mid)[0]
    if closed:
        s0 = gate_s.get(0)
        timing["start_finish"] = {"s": 0.0}
        if s0 is not None and min(s0, length - s0) > 0.5:
            timing["start_finish"] = {"s": round(s0, 3)}
            report.add("mapped", "/timing/start_finish", "AC_TIME_0_L/R gate midpoint")
        else:
            report.add("assumed", "/timing/start_finish", "AC lap starts at spline position 0")
        origin = timing["start_finish"]["s"]
        splits = sorted(s for i, s in gate_s.items() if i > 0)
        if splits:
            cuts = [origin] + [s for s in splits] + [origin + length]
            cuts = [c % length if i < len(cuts) - 1 else c for i, c in enumerate(cuts)]
            timing["sectors"] = [{"id": f"s{i + 1}", "s_start": round(cuts[i] % length, 3),
                                  "s_end": round(cuts[i + 1] % length, 3)} for i in range(len(cuts) - 1)]
            report.add("mapped", "/timing/sectors", f"{len(splits)} AC_TIME_n gates -> {len(splits) + 1} sectors")
    else:
        timing["start"] = {"s": 0.0}
        timing["finish"] = {"s": round(length, 3)}
        if gate_s:
            timing["split_points"] = [{"id": f"split_{i}", "s": round(s, 3)} for i, s in sorted(gate_s.items())
                                      if 0 < s < length and i > 0]

    # ── racing lines ─────────────────────────────────────────────────────────
    ex = fast.extras
    line = {"id": "fast_lane", "purpose": "race", "closed": closed,
            "points": cl["points"], "vehicle_class": "ac_ai_default"}
    if ex is not None:
        sp = np.clip(fast.extra("speed"), 0.0, 120.0)
        if float(np.max(fast.extra("speed"))) > 120.0:
            report.add("approximated", "/race/racing_lines/0/speed", "AI speeds above 120 m/s (placeholder values) clipped")
        line["speed"] = [round(float(v), 3) for v in sp]
        line["width"] = [round(float(a + b), 3) for a, b in zip(sl, sr)]
        report.add("mapped", "/race/racing_lines/0", "ai/fast_lane.ai points, speed (AI profile), width")
    race["racing_lines"] = [line]
    pit_ai = ai_dir / "pit_lane.ai"
    if pit_ai.exists() and pit_ai.stat().st_size > 20:
        try:
            pl = AiLine.read(pit_ai)
            pp = [ac_to_stj_xyz(*p) for p in pl.points]
            race["racing_lines"].append({"id": "pit_lane", "purpose": "pit", "closed": False,
                                         "points": [[round(float(v), 4) for v in p] for p in pp]})
            e_s, _, _ = project(np.array(pp[0]))
            x_s, _, _ = project(np.array(pp[-1]))
            if closed:
                timing["pit_lane"] = {"entry_s": round(e_s, 3), "exit_s": round(x_s, 3),
                                      "path": [[round(float(v), 4) for v in p] for p in pp]}
                report.add("mapped", "/timing/pit_lane", "ai/pit_lane.ai; entry/exit by projection onto the reference line")
            else:
                report.add("approximated", "/timing/pit_lane", "pit lane on an open track: kept only as race.racing_lines[pit]")
        except Exception as e:  # noqa: BLE001
            report.add("unsupported", "ai/pit_lane.ai", f"not read: {e}")

    # ── sectors / sections / DRS ─────────────────────────────────────────────
    sec_path = data / "sections.ini"
    if sec_path.exists():
        keep(sec_path)
        cp = read_ini(sec_path)
        secs = []
        for i, sec in enumerate(ini_sections(cp, "SECTION")):
            secs.append({"id": sid(sec.get("TEXT") or f"section_{i}", f"section_{i}") + f"_{i}",
                         "s_start": s_of(sec.get("IN")), "s_end": s_of(sec.get("OUT")),
                         "x_assetto_corsa": {"text": sec.get("TEXT", "")}})
        if secs:
            race["sections"] = secs
            report.add("approximated", "/race/sections",
                       "AC sections.ini are named sector/split marks, kept as race.sections "
                       "(not contiguous timing.sectors)")
    drs_path = data / "drs_zones.ini"
    if drs_path.exists():
        keep(drs_path)
        cp = read_ini(drs_path)
        zones = []
        for i, z in enumerate(ini_sections(cp, "ZONE")):
            zones.append({"id": f"drs_{i}", "detection_s": s_of(z.get("DETECTION")), "activation_s": s_of(z.get("START")),
                          "deactivation_s": s_of(z.get("END"))})
        if zones:
            timing["drs_zones"] = zones
            report.add("mapped", "/timing/drs_zones", f"{len(zones)} zones (fractions -> s)")

    # ── cameras ──────────────────────────────────────────────────────────────
    MAPPED_CAM = {"POSITION", "FORWARD", "MIN_FOV", "MAX_FOV", "NEAR_PLANE", "FAR_PLANE", "IN_POINT", "OUT_POINT", "NAME"}
    sets = []
    for cf in sorted(data.glob("cameras*.ini")):
        keep(cf)
        cp = read_ini(cf)
        cams = []
        for i, c in enumerate(ini_sections(cp, "CAMERA")):
            pos = ac_to_stj_xyz(*fvec(c.get("POSITION")))
            fw = fvec(c.get("FORWARD"))
            f_stj = ac_to_stj_xyz(*fw)
            look = [pos[k] + 100.0 * f_stj[k] for k in range(3)]
            mn, mx = fnum(c.get("MIN_FOV"), 40), fnum(c.get("MAX_FOV"), 40)
            cam = {"id": f"cam_{i}", "kind": "static", "position": [round(v, 4) for v in pos],
                   "look_at": [round(v, 4) for v in look],
                   "fov": [round(math.radians(mn), 6), round(math.radians(max(mx, mn)), 6)],
                   "near": fnum(c.get("NEAR_PLANE"), 1) or 1.0, "far": fnum(c.get("FAR_PLANE"), 35000) or 35000.0}
            inp, outp = fnum(c.get("IN_POINT"), -1), fnum(c.get("OUT_POINT"), -1)
            if inp >= 0 and outp >= 0:
                cam["active"] = {"s_start": s_of(inp), "s_end": s_of(outp)}
            # everything AC knows about the camera that STJ has no field for (exposure, DOF, shadow splits, spline,
            # UP vector, the original name and in/out points) stays with the camera
            cam["x_assetto_corsa"] = {k: v for k, v in c.items() if k not in MAPPED_CAM or k in ("NAME", "IN_POINT", "OUT_POINT")}
            cams.append(cam)
        hdr = dict(cp["HEADER"]) if cp.has_section("HEADER") else {}
        if cams:
            sets.append({"id": sid(cf.stem), "name": hdr.get("SET_NAME", cf.stem), "cameras": cams,
                         "x_assetto_corsa": {"file": cf.name, "header": hdr}})
    if sets:
        doc["cameras"] = {"sets": sets}
        report.add("mapped", "/cameras", f"{sum(len(x['cameras']) for x in sets)} cameras in {len(sets)} sets: position, view "
                                         "direction, fov, clip planes, active range; all other AC keys per camera in x_assetto_corsa")

    # ── environment / lighting ───────────────────────────────────────────────
    doc["environment"] = {"north_heading": round(math.pi / 2, 6)}  # +Y (= -z_AC) is north by assumption (AC has no geo data)
    keep(data / "lighting.ini")
    report.add("assumed", "/environment/north_heading", "AC has no north reference; +Y assumed north")
    from .lighting import import_lighting
    import_lighting(track_dir, layout, data, doc, report, raw)

    for name in ("groove.ini", "audio_sources.ini", "crew.ini", "startinglights.ini", "overlays.ini", "map.ini",
                 "camera_facing.ini", "ai_hints.ini"):
        keep(data / name)
    for ai_name in ("ai_hints.ini",):
        if (data / ai_name).exists():
            report.add("unsupported", f"data/{ai_name}",
                       "AC brake/danger hints are fractions of a vehicle-specific scale; kept verbatim in x_assetto_corsa")
    if (data / "ideal_line.ai").exists():
        report.add("dropped", "data/ideal_line.ai", "driving-aid line equals ai/fast_lane.ai in most tracks; not converted")
    if meshes:
        from .usd_export import export_kn5_meshes
        assets_dir, uri_base, textures = meshes
        export_kn5_meshes(track_dir, layout, doc, assets_dir, report, textures, uri_base)
    else:
        report.add("unsupported", "*.kn5", "meshes, textures, materials not exported (use --meshes)")

    doc["timing"] = timing
    doc["race"] = race
    if include_raw:
        doc["x_assetto_corsa"] = {"track": track_dir.name, "layout": layout, "files": raw,
                                  "kn5": [p.name for p in _models_for(track_dir, layout)]}
    doc["_metadata"].update({
        "description": f"Converted from Assetto Corsa track '{track_dir.name}'" + (f", layout '{layout}'" if layout else ""),
        "created": datetime.date.today().isoformat(),
        "licence": {"redistributable": False, "notes": "Derived from Assetto Corsa content; do not redistribute."},
        "provenance": {"type": "converted", "confidence": "medium", "detail": "ai/fast_lane.ai + data/*.ini + kn5 nodes",
                       "converted_from": [{"format": "Assetto Corsa", "tool": TOOL, "tool_version": __version__,
                                           "source": track_dir.name,
                                           "date": datetime.date.today().isoformat(), "lost": report.lost()}]},
    })
    return doc, report


def _find_ui(track_dir, layout):
    for p in ([track_dir / "ui" / layout / "ui_track.json"] if layout else []) + [track_dir / "ui" / "ui_track.json"]:
        if p.exists():
            return p
    return None


def _stj_tools():
    import os
    env = os.environ.get("STJ_REPO")
    for c in ([Path(env)] if env else []) + [Path(__file__).resolve().parents[2] / "stj-ref"]:
        if (c / "tools" / "stj_core.py").exists():
            return c / "tools"
    raise FileNotFoundError("STJ repository not found; set STJ_REPO to a clone of RFloEng/STJ-Standard-Track-Json")
