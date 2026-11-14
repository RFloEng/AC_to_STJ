"""ac-stj command line.

  python cli.py to-stj  <ac track folder> [-l LAYOUT] [-o OUTDIR]     AC -> STJ (all layouts when -l is omitted)
  python cli.py to-ac   <file.stj.json> -o OUTDIR [--name NAME]       STJ -> AC track folder
  python cli.py list    <ac tracks folder>
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ac_stj.ac_import import ac_to_stj, list_layouts  # noqa: E402


def cmd_to_stj(a):
    src = Path(a.track)
    out = Path(a.out or ".")
    out.mkdir(parents=True, exist_ok=True)
    layouts = [a.layout] if a.layout else list_layouts(src)
    for lay in layouts:
        meshes = (out / f"{src.name}_assets", f"{src.name}_assets/", not a.no_textures) if a.meshes else None
        try:
            doc, rep = ac_to_stj(src, lay, include_raw=not a.no_raw, meshes=meshes)
        except ValueError as e:
            print("skipped:", e)
            continue
        stem = src.name + (f"_{lay}" if lay else "")
        target = out / f"{stem}.stj.json"
        target.write_text(json.dumps(doc, indent=None if a.compact else 2), encoding="utf-8")
        rep.target = target.name
        (out / f"{stem}.conversion.json").write_text(json.dumps(rep.to_dict(), indent=2), encoding="utf-8")
        print(f"{target}  ({rep.counts()})")


def cmd_to_ac(a):
    from ac_stj.ac_export import stj_to_ac
    doc = json.loads(Path(a.stj).read_text(encoding="utf-8"))
    dest, rep = stj_to_ac(doc, Path(a.out), name=a.name, with_mesh=not a.no_mesh, base_dir=Path(a.stj).resolve().parent)
    print(f"{dest}  ({rep.counts()})")


def cmd_list(a):
    for p in sorted(Path(a.root).iterdir()):
        if p.is_dir():
            lay = list_layouts(p)
            if lay:
                print(p.name, [l or "(root)" for l in lay])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ac-stj", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("to-stj")
    p.add_argument("track")
    p.add_argument("-l", "--layout")
    p.add_argument("-o", "--out")
    p.add_argument("--no-raw", action="store_true", help="do not embed the original ini files in x_assetto_corsa")
    p.add_argument("--compact", action="store_true")
    p.add_argument("--meshes", action="store_true", help="also convert the kn5 meshes to USD (needs usd-core)")
    p.add_argument("--no-textures", action="store_true", help="with --meshes: do not copy textures")
    p.set_defaults(fn=cmd_to_stj)
    p = sub.add_parser("to-ac")
    p.add_argument("stj")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--name")
    p.add_argument("--no-mesh", action="store_true", help="skip the generated kn5 road mesh")
    p.set_defaults(fn=cmd_to_ac)
    p = sub.add_parser("list")
    p.add_argument("root")
    p.set_defaults(fn=cmd_list)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
