"""CSP Rally Copilot pacenotes (`*.rc-notes`)  <->  STJ `race.pacenotes` / `race.pacenote_sets`.

Rally Copilot (apps/lua/RallyCopilot) reads `*.rc-notes` from the layout's `data` folder and from the user's config folder, and
generates notes itself when there are none. A note is `RouteItem(type, modifier, pos, hints)`:
  type      1 TurnLeft, 2 TurnRight (modifier 1 = hairpin .. 6 = easiest), 3 KeepLeft, 4 KeepRight, 5 Straight (modifier = index of
            20 30 40 50 70 100 200 300 400 500 m), 6 Jump (1 big jump, 2 jump, 3 overcrest), 7 Narrows, 8 OverBridge,
            9 SurfaceType (1 tarmac, 2 gravel, 3 sand, 4 asphalt), 100 Extra
  pos       fraction of the track length (0..1)
  hints     1 Cut, 2 DoNotCut, 3 Long, 4 VeryLong, 5 Tightens, 6 Open, 7 Caution, 8 KeepLeft, 9 KeepRight
The text format is read from the app's source (PaceNotes.lua loads `{version = 3, items = [RouteItem(...)], metadata = {name, author}}`
through `stringify.tryParse`); no real .rc-notes file was available when this was written, so the reader is a tolerant regex
and the writer's exact layout is an assumption. The original file text is kept verbatim in x_assetto_corsa and written back
unchanged while the notes are unchanged.

Mapping is approximate: AC scales turns 1..6 with 1 = hairpin, STJ has a separate `hairpin` kind and severity 1 = slowest turn
short of a hairpin, so AC 2..6 map to severity 2..6 and severity 1 stays unused. Every item is also kept as it was in the note's
`x_assetto_corsa`, so the AC values come back exactly.
"""
import json
import re

STRAIGHTS = [20, 30, 40, 50, 70, 100, 200, 300, 400, 500]
HINT_MOD = {1: "cut", 2: "dont_cut", 3: "long", 4: "very_long", 5: "tightens", 6: "opens", 7: "caution",
            8: "x_keep_left", 9: "x_keep_right"}
MOD_HINT = {v: k for k, v in HINT_MOD.items()}
_ITEM = re.compile(r"RouteItem\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*([-+0-9.eE]+)\s*,\s*[{\[]([^}\]]*)[}\]]\s*\)")
_META = re.compile(r'"?(name|author)"?\s*[:=]\s*"((?:[^"\\]|\\.)*)"')


def parse_rc_notes(text):
    """-> (items [{type, modifier, pos, hints}], metadata {name, author}); empty list when nothing is recognised."""
    items = []
    for m in _ITEM.finditer(text):
        hints = [int(h) for h in re.findall(r"-?\d+", m.group(4))]
        items.append({"type": int(m.group(1)), "modifier": int(m.group(2)), "pos": float(m.group(3)), "hints": hints})
    tail = text[text.rfind("metadata"):] if "metadata" in text else ""
    meta = {k: json.loads('"' + v + '"') for k, v in _META.findall(tail)}
    return items, meta


def write_rc_notes(items, meta):
    items = sorted(items, key=lambda i: i["pos"])
    body = ",".join("RouteItem(%d,%d,%s,{%s})" % (i["type"], i["modifier"], repr(round(i["pos"], 6)),
                                                   ",".join(str(h) for h in i["hints"])) for i in items)
    return '{"version":3,"items":[%s],"metadata":%s}' % (body, json.dumps(meta or {"name": "STJ", "author": "ac-stj"}))


def item_to_note(it, length):
    t, mod, hints = it["type"], it["modifier"], it["hints"]
    note = {"s": round(it["pos"] * length, 3), "kind": "other"}
    mods = [HINT_MOD[h] for h in hints if h in HINT_MOD]
    if t in (1, 2):
        note["direction"] = "left" if t == 1 else "right"
        note["kind"] = "hairpin" if mod == 1 else "turn"
        if mod >= 2:
            note["severity"] = max(1, min(6, mod))
    elif t in (3, 4):
        note["direction"] = "left" if t == 3 else "right"
        note["kind"] = "other"
    elif t == 5:
        note["kind"] = "straight"
        if 1 <= mod <= len(STRAIGHTS):
            note["length"] = float(STRAIGHTS[mod - 1])
    elif t == 6:
        note["kind"] = "crest" if mod == 3 else "jump"
        if mod == 3:
            mods.append("over_crest")
    elif t == 7:
        note["kind"] = "narrows"
    elif t == 8:
        note["kind"] = "bridge"
    elif t == 9:
        note["kind"] = "surface_change"
    if mods:
        note["modifiers"] = list(dict.fromkeys(mods))
    note["x_assetto_corsa"] = {"type": t, "modifier": mod, "pos": it["pos"], "hints": list(hints)}
    return note


def note_to_item(note, length):
    x = note.get("x_assetto_corsa")
    if x and abs(x["pos"] * length - note["s"]) < 0.01:
        return {"type": x["type"], "modifier": x["modifier"], "pos": x["pos"], "hints": list(x["hints"])}
    kind, d = note["kind"], note.get("direction")
    hints = sorted({MOD_HINT[m] for m in note.get("modifiers", []) if m in MOD_HINT}, reverse=True)
    t, mod = 100, 0
    if kind in ("turn", "hairpin", "square", "acute", "twisty") and d:
        t = 1 if d == "left" else 2
        mod = 1 if kind in ("hairpin", "acute") else max(2, min(6, int(note.get("severity", 3))))
    elif kind == "straight":
        t = 5
        ln = note.get("length", 50.0)
        mod = 1 + min(range(len(STRAIGHTS)), key=lambda k: abs(STRAIGHTS[k] - ln))
    elif kind in ("jump", "crest", "bump"):
        t, mod = 6, 3 if kind == "crest" else 2
    elif kind == "narrows":
        t = 7
    elif kind == "bridge":
        t = 8
    elif kind == "surface_change":
        t, mod = 9, 1
    elif kind in ("keep_in", "keep_out") and d:
        t = 3 if d == "left" else 4
    return {"type": t, "modifier": mod, "pos": note["s"] / length, "hints": hints}


def import_pacenotes(data_dir, length, doc, report, raw):
    """data/*.rc-notes -> doc['race']['pacenotes'] (first file) and pacenote_sets (the rest); raw text kept in `raw`."""
    files = sorted(data_dir.glob("*.rc-notes"))
    sets = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        raw["data/" + f.name] = text
        items, meta = parse_rc_notes(text)
        if not items:
            report.add("unsupported", "data/" + f.name, "no RouteItem entries recognised; kept verbatim")
            continue
        notes = [item_to_note(i, length) for i in sorted(items, key=lambda i: i["pos"])]
        sets.append((f.stem, meta, notes))
    if not sets:
        return
    race = doc.setdefault("race", {})
    stem, meta, notes = sets[0]
    race["pacenotes"] = {"scale": "ascending_1_6", "notes": notes,
                         "x_assetto_corsa": {"file": stem + ".rc-notes", "metadata": meta}}
    if len(sets) > 1:
        race["pacenote_sets"] = [{"id": re.sub(r"[^a-z0-9_]+", "_", s.lower()).strip("_") or f"set_{i}", "scale": "ascending_1_6",
                                  "notes": n, "x_assetto_corsa": {"file": s + ".rc-notes", "metadata": m}}
                                 for i, (s, m, n) in enumerate(sets[1:], 1)]
    report.add("approximated", "/race/pacenotes",
               f"{sum(len(n) for _, _, n in sets)} Rally Copilot notes from {len(sets)} file(s): turns/hairpins/straights/jumps/"
               "narrows/bridges/surface changes and hints as modifiers; AC 1-6 turn scale -> STJ kinds+severity (see pacenotes.py); "
               "each item kept in x_assetto_corsa. Keep left/right and Extra items map to kind 'other'")


def export_pacenotes(doc, lay_data_dir, length, raw_data, report):
    race = doc.get("race", {})
    groups = []
    if race.get("pacenotes"):
        groups.append(race["pacenotes"])
    groups += race.get("pacenote_sets", [])
    for k, g in enumerate(groups):
        x = g.get("x_assetto_corsa") or {}
        fn = x.get("file") or ("stj.rc-notes" if k == 0 else f"stj_{k}.rc-notes")
        items = [note_to_item(n, length) for n in g["notes"]]
        old = raw_data.get(fn)
        if old is not None and [dict(i) for i in sorted(parse_rc_notes(old)[0], key=lambda i: i["pos"])] == \
                [dict(i) for i in sorted(items, key=lambda i: i["pos"])]:
            text = old
        else:
            text = write_rc_notes(items, x.get("metadata") or {"name": g.get("id", "STJ"), "author": "ac-stj"})
        (lay_data_dir / fn).write_text(text, encoding="utf-8")
        report.add("mapped", fn, f"{len(items)} pacenotes -> Rally Copilot file (layout of the file written from the app's source, "
                                 "not tested in game)")
