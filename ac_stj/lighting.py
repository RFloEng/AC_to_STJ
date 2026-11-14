"""Sun, CSP lights, geo position  <->  STJ environment.lighting / environment.geo_reference.

AC sources
  data/lighting.ini            SUN_PITCH_ANGLE, SUN_HEADING_ANGLE         -> environment.lighting.sun.direction (fixed rig, spec 8.3)
  extension/ext_config.ini     [LIGHT_n] (Custom Shaders Patch)           -> environment.lighting.lights[]
                               [BASIC] LATITUDE / LONGITUDE               -> environment.geo_reference
Convention assumed for the sun (not documented by Kunos): heading is a compass angle clockwise from AC +z,
pitch is the elevation above the horizon:  d_AC = (sin h cos p, sin p, cos h cos p).  The raw angles are kept in
x_assetto_corsa so nothing is lost when the assumption is wrong.
"""
import math
import re
from pathlib import Path

from .common import ac_to_stj_xyz, fnum, fvec, read_ini, stj_to_ac_xyz


# ── sun ──────────────────────────────────────────────────────────────────────
def sun_from_angles(pitch_deg, heading_deg):
    p, h = math.radians(pitch_deg), math.radians(heading_deg)
    d_ac = (math.sin(h) * math.cos(p), math.sin(p), math.cos(h) * math.cos(p))
    return [round(v, 6) for v in ac_to_stj_xyz(*d_ac)]


def angles_from_sun(direction):
    x, y, z = stj_to_ac_xyz(*direction)
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / n, y / n, z / n
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, y))))
    heading = math.degrees(math.atan2(x, z)) % 360.0
    return round(pitch, 2), round(heading, 2)


# ── colour helpers ───────────────────────────────────────────────────────────
def _srgb_lin(c):
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def rgb_to_cct(r, g, b):
    """McCamy approximation from sRGB 0..255; clamped to 1667-25000 K."""
    r, g, b = _srgb_lin(r), _srgb_lin(g), _srgb_lin(b)
    X = 0.4124 * r + 0.3576 * g + 0.1805 * b
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    Z = 0.0193 * r + 0.1192 * g + 0.9505 * b
    s = X + Y + Z
    if s <= 0:
        return 6500.0
    x, y = X / s, Y / s
    n = (x - 0.3320) / (0.1858 - y) if y != 0.1858 else 0.0
    return float(max(1667.0, min(25000.0, 449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33)))


def cct_to_rgb(k):
    """Tanner Helland approximation -> sRGB 0..255."""
    t = k / 100.0
    r = 255 if t <= 66 else max(0, min(255, 329.698727446 * (t - 60) ** -0.1332047592))
    g = (99.4708025861 * math.log(t) - 161.1195681661) if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255 if t >= 66 else (0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    return [int(round(max(0, min(255, v)))) for v in (r, g, b)]


CD_PER_BRIGHTNESS = 50.0     # estimate: CSP LIGHT brightness (4th COLOR value) -> candela


def _kind(desc):
    d = (desc or "").lower()
    if "street" in d or "lamp" in d:
        return "street"
    if "flood" in d or "projector" in d:
        return "floodlight"
    if "pit" in d:
        return "pit"
    if "start" in d:
        return "start_lights"
    return "other"


# ── import ───────────────────────────────────────────────────────────────────
def ext_config_paths(track_dir, layout):
    base = Path(track_dir) / "extension"
    out = []
    if base.is_dir():
        for p in sorted(base.glob("*.ini")):
            n = p.stem.lower()
            if n in ("ext_config",) or (layout and layout.lower() in n):
                out.append(p)
    return out


def import_lighting(track_dir, layout, data_dir, doc, report, raw):
    """Fill doc['environment'] lighting/geo from lighting.ini and the CSP ext_config.ini. Returns nothing."""
    env = doc.setdefault("environment", {})
    lighting = env.setdefault("lighting", {})
    lp = Path(data_dir) / "lighting.ini"
    if lp.exists():
        cp = read_ini(lp)
        if cp.has_section("LIGHTING"):
            pitch = fnum(cp["LIGHTING"].get("SUN_PITCH_ANGLE"), 35)
            head = fnum(cp["LIGHTING"].get("SUN_HEADING_ANGLE"), 0)
            lighting["sun"] = {"direction": sun_from_angles(pitch, head),
                               "x_assetto_corsa": {"pitch_deg": pitch, "heading_deg": head,
                                                   "convention": "heading clockwise from AC +z (assumed)"}}
            report.add("approximated", "/environment/lighting/sun/direction",
                       "SUN_PITCH/HEADING -> fixed sun vector; heading convention assumed (raw angles kept in x_assetto_corsa)")
    for p in ext_config_paths(track_dir, layout):
        text = p.read_text(encoding="utf-8", errors="replace")
        raw[str(p.relative_to(track_dir)).replace("\\", "/")] = text
        cp = read_ini(p)
        if cp.has_section("BASIC"):
            lat, lon = fnum(cp["BASIC"].get("LATITUDE"), None), fnum(cp["BASIC"].get("LONGITUDE"), None)
            if lat is not None and lon is not None and (lat or lon):
                env.setdefault("geo_reference", {}).update({"latitude": lat, "longitude": lon})
                report.add("mapped", "/environment/geo_reference", "CSP [BASIC] LATITUDE/LONGITUDE")
        lights = []
        for sec in cp.sections():
            m = re.fullmatch(r"LIGHT_(\d+)", sec, re.I)
            if not m:
                continue
            s = cp[sec]
            col = fvec(s.get("COLOR"), 4)
            spot = fnum(s.get("SPOT"), 0)
            pos = ac_to_stj_xyz(*fvec(s.get("POSITION")))
            cond = (s.get("CONDITION") or "").upper()
            L = {"id": f"light_{int(m.group(1))}", "kind": _kind(s.get("DESCRIPTION")),
                 "type": "spot" if spot > 0 else "point", "position": [round(v, 3) for v in pos],
                 "intensity": round(max(col[3], 0.0) * CD_PER_BRIGHTNESS, 3),
                 "color_temperature": round(rgb_to_cct(col[0], col[1], col[2]), 0),
                 "schedule": "night" if "NIGHT" in cond else ("always" if not cond else "event"),
                 "x_assetto_corsa": {k: v for k, v in s.items()}}
            if fnum(s.get("RANGE")) > 0:
                L["range"] = fnum(s.get("RANGE"))
            if spot > 0:
                d = ac_to_stj_xyz(*fvec(s.get("DIRECTION"), 3))
                L["direction"] = [round(v, 5) for v in d]
                outer = math.radians(spot) / 2.0
                L["outer_cone_angle"] = round(outer, 5)
                L["inner_cone_angle"] = round(outer * max(0.0, min(1.0, fnum(s.get("SPOT_SHARPNESS"), 0.5))), 5)
            lights.append(L)
        if lights:
            lights.sort(key=lambda L: int(L["id"].split("_")[1]))
            lighting.setdefault("lights", []).extend(lights)
            report.add("approximated", "/environment/lighting/lights",
                       f"{len(lights)} CSP [LIGHT_n] from {p.name}: position, spot cone, range, night schedule mapped; "
                       f"colour -> colour temperature, brightness x{CD_PER_BRIGHTNESS:g} -> cd (estimates); "
                       "every original key kept per light in x_assetto_corsa")
    if not lighting:
        env.pop("lighting")


# ── export ───────────────────────────────────────────────────────────────────
def lighting_ini(sun, raw_text=None):
    pitch, head = angles_from_sun(sun["direction"])
    x = sun.get("x_assetto_corsa", {})
    if x.get("pitch_deg") is not None and angles_from_sun(sun_from_angles(x["pitch_deg"], x["heading_deg"])) == (pitch, head):
        pitch, head = x["pitch_deg"], x["heading_deg"]       # unchanged: write the original numbers back
    text = raw_text or "[LIGHTING]\n"
    for key, val in (("SUN_PITCH_ANGLE", pitch), ("SUN_HEADING_ANGLE", head)):
        val = int(val) if float(val).is_integer() else val
        if re.search(rf"^{key}=", text, re.M):
            text = re.sub(rf"^{key}=.*$", f"{key}={val}", text, flags=re.M)
        else:
            text = text.rstrip("\n") + f"\n{key}={val}\n"
    return text


def ext_config_text(doc, raw_text=None):
    """CSP ext_config.ini: raw file with its LIGHT_n sections replaced by the STJ lights."""
    env = doc.get("environment", {})
    lights = env.get("lighting", {}).get("lights", [])
    geo = env.get("geo_reference", {})
    text = raw_text or ""
    if raw_text:
        text = re.sub(r"(?ms)^\[LIGHT_\d+\].*?(?=^\[|\Z)", "", raw_text)
    if geo.get("latitude") is not None and not re.search(r"^\[BASIC\]", text, re.M):
        text = text.rstrip("\n") + f"\n\n[BASIC]\nLATITUDE={geo['latitude']}\nLONGITUDE={geo['longitude']}\n"
    out = [text.rstrip("\n") + "\n"] if text.strip() else []
    for i, L in enumerate(lights):
        sec = dict(L.get("x_assetto_corsa") or {})
        px, py, pz = stj_to_ac_xyz(*L["position"])
        sec["POSITION"] = f"{px:.2f}, {py:.2f}, {pz:.2f}"
        if "COLOR" not in sec:
            r, g, b = cct_to_rgb(L.get("color_temperature", 3000.0))
            sec["COLOR"] = f"{r}, {g}, {b}, {L.get('intensity', 300.0) / CD_PER_BRIGHTNESS:g}"
        if L.get("range") and "RANGE" not in sec:
            sec["RANGE"] = f"{L['range']:g}"
        if L["type"] == "spot":
            if "SPOT" not in sec:
                sec["SPOT"] = f"{math.degrees(L.get('outer_cone_angle', math.radians(60))) * 2:g}"
            dx, dy, dz = stj_to_ac_xyz(*L.get("direction", [0, 0, -1]))
            sec["DIRECTION"] = f"{dx:.5g}, {dy:.5g}, {dz:.5g}"
        sec.setdefault("DESCRIPTION", L.get("kind", "light"))
        if "CONDITION" not in sec and L.get("schedule", "night") == "night":
            sec["CONDITION"] = "NIGHT_SMOOTH"
        out.append(f"[LIGHT_{i}]\n" + "\n".join(f"{k}={v}" for k, v in sec.items()) + "\n")
    return "\n".join(out)
