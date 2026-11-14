"""kn5 meshes -> OpenUSD (STJ geometry.meshes).

* Collision meshes (AC names ``01ROAD001``, ``1GRASS`` ...: leading digits + surface KEY) go to a ``*.collision.usdc``
  whose materials are named ``STJ__surface__<id>`` (spec 6.3 level 1), so the physics follows from the file.
* Everything else goes to a ``*.visual.usdc`` with UsdPreviewSurface materials; the AC shader and its raw
  parameters stay in the material's customData under ``x_assetto_corsa`` (spec 7.7, fidelity rule 1).
* Frame: AC (x, y, z) -> STJ (x, -z, y), node transforms applied, UV v flipped, winding checked against the
  stored vertex normals.  Textures are copied unchanged (DDS is kept, spec 8.7).
"""
import re
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from .ac_import import _models_for, classify, sid
from .ac_export import CLASS_DEFAULT
from .kn5 import Kn5Data, read_kn5

COLLISION_RX = re.compile(r"^\d{1,2}[A-Z_\-]")


def _prim_name(name, used):
    n = re.sub(r"[^A-Za-z0-9_]", "_", name) or "mesh"
    if n[0].isdigit():
        n = "m" + n
    base, i = n, 1
    while n in used:
        i += 1
        n = f"{base}_{i}"
    used.add(n)
    return n


def surface_for(mesh_name, lib_keys):
    """Library id for an AC collision mesh name ('01ROAD003' -> key ROAD), longest KEY prefix wins."""
    core = re.sub(r"^\d+", "", mesh_name).upper()
    best = None
    for key, sid_ in lib_keys.items():
        k = key.upper()
        if core.startswith(k) and (best is None or len(k) > len(best[0])):
            best = (k, sid_)
    return best[1] if best else None


def _world(mesh, world):
    """AC-frame mesh -> STJ-frame arrays (points, normals, uvs, triangles)."""
    m = np.array(world, dtype=np.float64)
    p = mesh["positions"].astype(np.float64) @ m[:3, :3] + m[3, :3]
    n = mesh["normals"].astype(np.float64) @ m[:3, :3]
    f = lambda a: np.stack([a[:, 0], -a[:, 2], a[:, 1]], axis=1)
    p, n = f(p), f(n)
    uv = mesh["uvs"].astype(np.float64).copy()
    uv[:, 1] = 1.0 - uv[:, 1]
    tri = mesh["triangles"].astype(np.int64)
    if len(tri):
        a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
        fn = np.cross(b - a, c - a)
        vn = n[tri[:, 0]] + n[tri[:, 1]] + n[tri[:, 2]]
        if float(np.sum(fn * vn)) < 0:
            tri = tri[:, ::-1]
    nl = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(nl > 0, nl, 1.0)
    return p.astype(np.float32), n.astype(np.float32), uv.astype(np.float32), np.ascontiguousarray(tri)


def _new_stage(path):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/track")
    stage.SetDefaultPrim(root.GetPrim())
    return stage


def _write_mesh(stage, path, p, n, uv, tri, double_sided=False, flags=None):
    mesh = UsdGeom.Mesh.Define(stage, path)
    if flags:
        # AC per-mesh render flags and LOD range stay with the prim; the ones USD has a native word for are also set
        mesh.GetPrim().SetCustomDataByKey("x_assetto_corsa", {k: (float(v) if isinstance(v, float) else v) for k, v in flags.items()})
        if not flags.get("visible", True):
            mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        if not flags.get("cast_shadows", True):
            mesh.GetPrim().CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(p))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tri), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tri.reshape(-1).astype(np.int32)))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(n))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    pv.Set(Vt.Vec2fArray.FromNumpy(uv))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(double_sided)
    mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([p.min(0), p.max(0)], dtype=np.float32)))
    return mesh


def _flags(m, name=""):
    f = {"cast_shadows": m["cast_shadows"], "visible": m["visible"], "transparent": m["transparent"], "layer": int(m["layer"]),
         "lod_in": float(m["lod_in"]), "lod_out": float(m["lod_out"]), "node": name or None}
    if name.startswith("AC_POBJECT"):
        f["physics_object"] = True     # AC movable prop: exported as a static mesh, its behaviour is not described
    return {k: v for k, v in f.items() if v is not None}


def _tex_asset(name, tex_rel):
    return Sdf.AssetPath(f"{tex_rel}/{name}")


def _make_visual_material(stage, mat, used_names, tex_rel):
    name = _prim_name(mat["name"], used_names)
    m = UsdShade.Material.Define(stage, f"/track/Materials/{name}")
    sh = UsdShade.Shader.Define(stage, f"/track/Materials/{name}/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    pr, tx = mat["props"], mat["textures"]
    diff = tx.get("txDiffuse")
    reader = None
    if diff or tx.get("txNormal"):
        reader = UsdShade.Shader.Define(stage, f"/track/Materials/{name}/st")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    def tex(node, fname, colorspace, scale=None, bias=None):
        t = UsdShade.Shader.Define(stage, f"/track/Materials/{name}/{node}")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(_tex_asset(fname, tex_rel))
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.GetOutput("result"))
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(colorspace)
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        if scale:
            t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
            t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
        t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        t.CreateOutput("a", Sdf.ValueTypeNames.Float)
        return t

    if diff:
        t = tex("diffuse", diff, "sRGB")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(t.GetOutput("rgb"))
        if mat["alpha_tested"]:
            sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(t.GetOutput("a"))
            ref = pr.get("ksAlphaRef", 0.0)
            sh.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(float(ref) if 0 < ref < 1 else 0.5)
        elif mat["alpha_blend"]:
            sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(t.GetOutput("a"))
    else:
        d = float(pr.get("ksDiffuse", 0.5))
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(d, d, d))
    if tx.get("txNormal"):
        t = tex("normal", tx["txNormal"], "raw", scale=(2, 2, 2, 1), bias=(-1, -1, -1, 0))
        sh.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(t.GetOutput("rgb"))
    notes = []
    spec = float(pr.get("ksSpecular", 0.0))
    maps_tex = tx.get("txMaps")
    sh.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
    sh.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(spec, spec, spec))
    if maps_tex and reader is not None:
        t = tex("maps", maps_tex, "raw", scale=(1, -1, 1, 1), bias=(0, 1, 0, 0))   # roughness = 1 - gloss(G)
        t.CreateOutput("g", Sdf.ValueTypeNames.Float)
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(t.GetOutput("g"))
        notes.append("txMaps: G read as gloss (roughness = 1 - G); R (specular) and B not mapped - assumption, see shader customData")
    else:
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(np.clip(1.0 - min(pr.get("ksSpecularEXP", 1.0), 250.0) / 250.0, 0.15, 1.0)))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    em = float(pr.get("ksEmissive", 0.0))
    if em > 0:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(em, em, em))
    sh.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    m.CreateSurfaceOutput().ConnectToSource(sh.GetOutput("surface"))
    if mat["shader"].startswith("ksMultilayer") and tx.get("txMask"):
        notes.append(_multilayer_mtlx(stage, m, name, mat))
    m.GetPrim().SetCustomDataByKey("x_assetto_corsa", {
        **({"notes": notes} if notes else {}),
        "shader": mat["shader"], "alpha_blend": int(mat["alpha_blend"]), "alpha_tested": bool(mat["alpha_tested"]),
        "props": {k: float(v) for k, v in pr.items()}, "textures": dict(tx)})
    return m, (mat["alpha_tested"] or bool(mat["alpha_blend"]))


def _multilayer_mtlx(stage, mat_prim, name, mat):
    """ksMultilayer*: four tiled detail layers (txDetailR/G/B/A, tiling multR..A) weighted by the RGBA channels of txMask,
    times magicMult, as an OpenPBR MaterialX network on outputs:mtlx:surface (spec 7.7 rule 2). Structure follows the
    MaterialX 1.39 standard library; it has not been rendered."""
    tx, pr = mat["textures"], mat["props"]
    base = f"/track/Materials/{name}/mtlx"
    UsdGeom.Scope.Define(stage, base)

    def node(nid, name_, out_type):
        n = UsdShade.Shader.Define(stage, f"{base}/{name_}")
        n.CreateIdAttr(nid)
        n.CreateOutput("out", out_type)
        return n

    def inp(n, key, vtype, value=None, src=None):
        i = n.CreateInput(key, vtype)
        if src is not None:
            i.ConnectToSource(src)
        else:
            i.Set(value)

    V2, C3, C4, F = Sdf.ValueTypeNames.Float2, Sdf.ValueTypeNames.Color3f, Sdf.ValueTypeNames.Color4f, Sdf.ValueTypeNames.Float
    uv = node("ND_geompropvalue_vector2", "uv", V2)
    inp(uv, "geomprop", Sdf.ValueTypeNames.String, "st")
    mask = node("ND_image_color4", "mask", C4)
    inp(mask, "file", Sdf.ValueTypeNames.Asset, Sdf.AssetPath("../textures/" + tx["txMask"]))
    inp(mask, "texcoord", V2, src=uv.GetOutput("out"))
    sep = UsdShade.Shader.Define(stage, f"{base}/sep")
    sep.CreateIdAttr("ND_separate4_color4")
    inp(sep, "in", C4, src=mask.GetOutput("out"))
    for ch in ("outr", "outg", "outb", "outa"):
        sep.CreateOutput(ch, F)
    total = None
    for ch, key, mult in (("outr", "txDetailR", "multR"), ("outg", "txDetailG", "multG"),
                          ("outb", "txDetailB", "multB"), ("outa", "txDetailA", "multA")):
        if key not in tx:
            continue
        tile = node("ND_multiply_vector2FA", f"tile_{ch}", V2)
        inp(tile, "in1", V2, src=uv.GetOutput("out"))
        inp(tile, "in2", F, float(pr.get(mult, 1.0)))
        img = node("ND_image_color3", f"img_{ch}", C3)
        inp(img, "file", Sdf.ValueTypeNames.Asset, Sdf.AssetPath("../textures/" + tx[key]))
        inp(img, "texcoord", V2, src=tile.GetOutput("out"))
        w = node("ND_multiply_color3FA", f"w_{ch}", C3)
        inp(w, "in1", C3, src=img.GetOutput("out"))
        inp(w, "in2", F, src=sep.GetOutput(ch))
        if total is None:
            total = w
        else:
            add = node("ND_add_color3", f"sum_{ch}", C3)
            inp(add, "in1", C3, src=total.GetOutput("out"))
            inp(add, "in2", C3, src=w.GetOutput("out"))
            total = add
    if total is None:
        return "ksMultilayer: no detail textures"
    scaled = node("ND_multiply_color3FA", "magic", C3)
    inp(scaled, "in1", C3, src=total.GetOutput("out"))
    inp(scaled, "in2", F, float(pr.get("magicMult", 1.0)))
    surf = UsdShade.Shader.Define(stage, f"{base}/openpbr")
    surf.CreateIdAttr("ND_open_pbr_surface_surfaceshader")
    inp(surf, "base_color", C3, src=scaled.GetOutput("out"))
    inp(surf, "specular_roughness", F, float(np.clip(1.0 - min(pr.get("ksSpecularEXP", 1.0), 250.0) / 250.0, 0.15, 1.0)))
    surf.CreateOutput("out", Sdf.ValueTypeNames.Token)
    out = mat_prim.CreateOutput("mtlx:surface", Sdf.ValueTypeNames.Token)
    out.ConnectToSource(surf.ConnectableAPI(), "out")
    return "ksMultilayer: MaterialX layer blend by txMask RGBA written on outputs:mtlx:surface (not rendered)"


def _make_surface_material(stage, sid_, cache, key=None):
    if sid_ not in cache:
        m = UsdShade.Material.Define(stage, f"/track/Materials/STJ__surface__{sid_}")
        sh = UsdShade.Shader.Define(stage, f"/track/Materials/STJ__surface__{sid_}/surface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.3, 0.3, 0.3))
        sh.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        m.CreateSurfaceOutput().ConnectToSource(sh.GetOutput("surface"))
        m.GetPrim().SetCustomDataByKey("x_assetto_corsa", {"surface_key": key or sid_.upper()})
        cache[sid_] = m
    return cache[sid_]


def export_kn5_meshes(track_dir, layout, doc, assets_dir, report, textures=True, uri_base=""):
    """Convert the layout's kn5 files into USD next to the STJ file and register them in doc['geometry'].

    ``assets_dir`` is where meshes/ and textures/ go; ``uri_base`` is its path relative to the STJ document.
    Returns the list of geometry.meshes entries added.
    """
    track_dir, assets_dir = Path(track_dir), Path(assets_dir)
    (assets_dir / "meshes").mkdir(parents=True, exist_ok=True)
    lib = doc.setdefault("surfaces", {}).setdefault("library", {})
    keys = {(s.get("name") or k): k for k, s in lib.items()}
    default_surface = doc.get("surfaces", {}).get("default") or (next(iter(lib)) if lib else None)
    entries, unmatched, builtin = [], set(), set()
    stats = {"visual": 0, "collision": 0, "tris": 0}

    for kn5 in _models_for(track_dir, layout):
        stem = sid(kn5.stem)
        vis_path = assets_dir / "meshes" / f"{track_dir.name}__{stem}.visual.usdc"
        col_path = assets_dir / "meshes" / f"{track_dir.name}__{stem}.collision.usdc"
        if not (vis_path.exists() or col_path.exists()):
            data = Kn5Data()
            try:
                _, root = read_kn5(kn5, keep_geometry=True, data=data)
            except Exception as e:  # noqa: BLE001
                report.add("unsupported", kn5.name, f"kn5 geometry not read: {e}")
                continue
            vis_stage = col_stage = None
            used_v, used_c, used_m, col_mats = set(), set(), set(), {}
            mat_map = {}
            tex_needed = set()
            for node, world in root.walk():
                m = node.mesh
                if not m or "positions" not in m or node.name.startswith("AC_") or len(m["triangles"]) == 0:
                    continue
                p, n, uv, tri = _world(m, world)
                if COLLISION_RX.match(node.name):
                    if col_stage is None:
                        col_stage = _new_stage(col_path)
                    sid_ = surface_for(node.name, keys)
                    if sid_ is None:
                        # AC has built-in surfaces (ROAD, GRASS, SAND, WALL, ...) that need no surfaces.ini entry
                        key = re.sub(r"\d+$", "", re.sub(r"^\d+", "", node.name)).upper()
                        sid_ = sid(key, "surface")
                        if sid_ not in lib:
                            cls = classify(key)
                            kd = CLASS_DEFAULT.get(cls, CLASS_DEFAULT["other"])
                            lib[sid_] = {"name": key, "class": cls,
                                         "mf_scaling": {"LMUX": kd[1], "LMUY": kd[1], "_est": True},
                                         "track_limits": "inside" if kd[2] else "outside"}
                            doc.setdefault("surfaces", {}).setdefault("library", lib)
                            keys[key] = sid_
                            builtin.add(key)
                    mesh = _write_mesh(col_stage, f"/track/collision/{_prim_name(node.name, used_c)}", p, n, uv, tri, flags=_flags(m))
                    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(_make_surface_material(col_stage, sid_, col_mats, lib.get(sid_, {}).get("name")))
                    stats["collision"] += 1
                else:
                    if vis_stage is None:
                        vis_stage = _new_stage(vis_path)
                    mi = next((i for i, mt in enumerate(data.materials) if mt["name"] == m["material"]), None)
                    ds = False
                    if mi is not None:
                        if mi not in mat_map:
                            mat_map[mi] = _make_visual_material(vis_stage, data.materials[mi], used_m,
                                                                "../textures" if textures else "../textures")
                            tex_needed.update(data.materials[mi]["textures"].values())
                        mat, ds = mat_map[mi]
                    mesh = _write_mesh(vis_stage, f"/track/visual/{_prim_name(node.name, used_v)}", p, n, uv, tri, ds,
                                       flags=_flags(m, node.name))
                    if mi is not None:
                        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
                    stats["visual"] += 1
                stats["tris"] += len(tri)
            for st in (vis_stage, col_stage):
                if st is not None:
                    st.GetRootLayer().Save()
            if textures and tex_needed:
                (assets_dir / "textures").mkdir(exist_ok=True)
                for t in tex_needed:
                    if t in data.textures and not (assets_dir / "textures" / t).exists():
                        (assets_dir / "textures" / t).write_bytes(data.texture_bytes(t))
        if col_path.exists():
            cstage = Usd.Stage.Open(str(col_path))
            for prim in cstage.GetPrimAtPath("/track/Materials").GetChildren():
                m = re.match(r"STJ__surface__(.+)", prim.GetName())
                if m and m.group(1) not in lib:
                    key = (prim.GetCustomDataByKey("x_assetto_corsa") or {}).get("surface_key", m.group(1).upper())
                    cls = classify(key)
                    kd = CLASS_DEFAULT.get(cls, CLASS_DEFAULT["other"])
                    lib[m.group(1)] = {"name": key, "class": cls,
                                       "mf_scaling": {"LMUX": kd[1], "LMUY": kd[1], "_est": True},
                                       "track_limits": "inside" if kd[2] else "outside"}
                    builtin.add(key)
        for path, role in ((vis_path, "visual"), (col_path, "collision")):
            if path.exists():
                entries.append({"id": f"{stem}_{role}", "uri": f"{uri_base}meshes/{path.name}", "roles": [role]})
    if builtin:
        report.add("assumed", "/surfaces/library", f"AC built-in surfaces {sorted(builtin)} (used by collision meshes, absent "
                                                   "from surfaces.ini) added with estimated friction")
    if entries:
        doc.setdefault("geometry", {}).setdefault("meshes", []).extend(entries)
        report.add("mapped", "/geometry/meshes",
                   f"{len(entries)} USD files from kn5 ({stats['visual']} visual + {stats['collision']} collision meshes, "
                   f"{stats['tris']:,} triangles; frame AC->STJ, UV v flipped)")
        report.add("approximated", "/geometry/meshes/materials",
                   "ks* shaders mapped to UsdPreviewSurface (diffuse, normal, alpha cut-out/blend, constant specular); "
                   "multi-map, detail, fresnel and damage layers are kept only in customData x_assetto_corsa")
        if not textures:
            report.add("dropped", "/geometry/meshes/textures", "textures not copied (--no-textures)")
    return entries
