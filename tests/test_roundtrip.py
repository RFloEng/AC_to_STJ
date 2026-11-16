"""AC -> STJ -> AC -> STJ round trip on an installed track (skipped when AC is not installed)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ac_stj.ac_import import ac_to_stj  # noqa: E402
from ac_stj.ac_export import stj_to_ac  # noqa: E402

AC = Path(os.environ.get("AC_TRACKS", "D:/Steam/steamapps/common/assettocorsa/content/tracks"))


@unittest.skipUnless((AC / "ks_vallelunga").exists(), "Assetto Corsa not installed")
class RoundTrip(unittest.TestCase):
    def test_vallelunga(self):
        a, _ = ac_to_stj(AC / "ks_vallelunga", "classic_circuit", include_raw=False)
        with tempfile.TemporaryDirectory() as d:
            root, _ = stj_to_ac(ac_to_stj(AC / "ks_vallelunga", "classic_circuit")[0], d)
            b, _ = ac_to_stj(root, "classic_circuit", include_raw=False)
        self.assertAlmostEqual(a["reference"]["length"], b["reference"]["length"], delta=2.0)
        pa = np.array(a["reference"]["centerline"]["points"])
        pb = np.array(b["reference"]["centerline"]["points"])
        dev = max(np.min(np.linalg.norm(pa - p, axis=1)) for p in pb[::10])
        self.assertLess(dev, 2.0)
        self.assertEqual(len(a["race"]["grid"]), len(b["race"]["grid"]))
        self.assertEqual(len(a["race"]["pit_boxes"]), len(b["race"]["pit_boxes"]))
        for ga, gb in zip(a["race"]["grid"], b["race"]["grid"]):
            self.assertLess(np.linalg.norm(np.array(ga["pose"]["position"]) - gb["pose"]["position"]), 0.01)


@unittest.skipUnless((AC / "ks_vallelunga").exists(), "Assetto Corsa not installed")
class MeshRoundTrip(unittest.TestCase):
    def test_meshes(self):
        from ac_stj.kn5 import Kn5Data, read_kn5
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            doc, _ = ac_to_stj(AC / "ks_vallelunga", "club_circuit", meshes=(d / "assets", "assets/", False))
            self.assertTrue(any("collision" in m["roles"] for m in doc["geometry"]["meshes"]))
            root, _ = stj_to_ac(doc, d / "ac", base_dir=d)
            import re
            src = 0
            for f in ("1.kn5", "4.kn5"):  # physics meshes also live in the layout kn5
                _, r = read_kn5(AC / "ks_vallelunga" / f, keep_geometry=True)
                src += sum(len(n.mesh["triangles"]) for n, _ in r.walk()
                           if n.mesh and "triangles" in n.mesh and re.match(r"^\d{1,2}[A-Z_\-]", n.name))
            _, r2 = read_kn5(root / "1.kn5", keep_geometry=True)
            t = lambda r: sum(len(n.mesh["triangles"]) for n, _ in r.walk() if n.mesh and "triangles" in n.mesh)
            self.assertGreater(t(r2), 0)
            self.assertAlmostEqual(t(r2), src, delta=src * 0.02)


if __name__ == "__main__":
    unittest.main()


class PacenotesRoundTrip(unittest.TestCase):
    """Rally Copilot .rc-notes <-> race.pacenotes on a synthetic file (the text layout is read from the app's source)."""
    TEXT = ('{"version":3,"items":[RouteItem(2,3,0.5,{7,3}),RouteItem(1,1,0.1,{}),RouteItem(5,6,0.25,{}),'
            'RouteItem(6,3,0.6,{}),RouteItem(3,0,0.7,{})],"metadata":{"name":"Test","author":"me"}}')

    def test_notes(self):
        import tempfile
        from ac_stj.pacenotes import export_pacenotes, import_pacenotes, parse_rc_notes

        class Rep:
            def add(self, *a):
                pass

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "stage.rc-notes").write_text(self.TEXT, encoding="utf-8")
            doc, raw = {}, {}
            import_pacenotes(d, 2000.0, doc, Rep(), raw)
            notes = doc["race"]["pacenotes"]["notes"]
            self.assertEqual([n["s"] for n in notes], [200.0, 500.0, 1000.0, 1200.0, 1400.0])
            self.assertEqual(notes[0]["kind"], "hairpin")
            self.assertEqual(notes[1]["kind"], "straight")
            self.assertEqual(notes[2]["modifiers"], ["caution", "long"])
            out = d / "out"
            out.mkdir()
            export_pacenotes(doc, out, 2000.0, {k.split("/")[-1]: v for k, v in raw.items()}, Rep())
            self.assertEqual((out / "stage.rc-notes").read_text(encoding="utf-8"), self.TEXT)      # unchanged -> verbatim
            doc["race"]["pacenotes"]["notes"][0]["s"] = 300.0                                          # edited -> regenerated
            export_pacenotes(doc, out, 2000.0, {k.split("/")[-1]: v for k, v in raw.items()}, Rep())
            items = parse_rc_notes((out / "stage.rc-notes").read_text(encoding="utf-8"))[0]
            self.assertAlmostEqual(items[0]["pos"], 0.15)
            self.assertEqual(len(items), 5)
