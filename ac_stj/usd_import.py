"""STJ geometry.meshes (OpenUSD) -> kn5 builders (visual + physics).  Inverse of usd_export."""
import re
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdShade

LIMIT = 60000


def _to_ac(p):
    return np.stack([p[:, 0], p[:, 2], -p[:, 1]], axis=1)


def _mesh_arrays(mesh, xf):
    """Triangulated arrays (points, normals, uvs, triangles) in STJ frame; unwelded when data is per corner."""
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64).reshape(-1, 3)
    m = np.array(xf).reshape(4, 4)
    pts = pts @ m[:3, :3] + m[3, :3]
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    idx = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    if np.all(counts == 3):
        corners = (starts[:, None] + np.arange(3)[None, :])
    else:
        corners = np.array([(st, st + k, st + k + 1) for st, c in zip(starts, counts) for k in range(1, c - 1)],
                           dtype=np.int64).reshape(-1, 3)
    vidx = idx[corners]
    nattr = mesh.GetNormalsAttr().Get()
    nint = mesh.GetNormalsInterpolation() if nattr is not None else None
    pv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    uvs = np.array(pv.Get(), dtype=np.float64).reshape(-1, 2) if pv and pv.Get() is not None else None
    uint = pv.GetInterpolation() if uvs is not None else None
    if nint == "faceVarying" or uint == "faceVarying":
        flat = corners.reshape(-1)
        P = pts[idx[flat]]
        tri = np.arange(len(flat)).reshape(-1, 3)
        N = None
        if nattr is not None:
            na = np.array(nattr, dtype=np.float64).reshape(-1, 3)
            N = na[flat] if nint == "faceVarying" else na[idx[flat]]
        U = None
        if uvs is not None:
            U = uvs[flat] if uint == "faceVarying" else uvs[idx[flat]]
        pts = P
    else:
        N = np.array(nattr, dtype=np.float64).reshape(-1, 3) if nattr is not None else None
        U = uvs
        tri = vidx
    if N is None:
        a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
        fn = np.cross(b - a, c - a)
        N = np.zeros_like(pts)
        for k in range(3):
            np.add.at(N, tri[:, k], fn)
        ln = np.linalg.norm(N, axis=1, keepdims=True)
        N = N / np.where(ln > 0, ln, 1)
    if U is None:
        U = np.zeros((len(pts), 2))
    return pts, N, U, tri


def _material_spec(prim, copy_tex):
    """(name, shader, props, textures, alpha_blend, alpha_tested) for the bound material, or None."""
    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not mat:
        return None
    name = mat.GetPrim().GetName()
    cd = mat.GetPrim().GetCustomDataByKey("x_assetto_corsa")
    if cd and "shader" in cd:
        textures = dict(cd.get("textures", {}))
        for tn in textures.values():
            copy_tex(tn)
        return (name, cd["shader"], dict(cd.get("props", {})), textures,
                int(cd.get("alpha_blend", 0)), bool(cd.get("alpha_tested", False)))
    props = {"ksAmbient": 0.4, "ksDiffuse": 0.5, "ksSpecular": 0.0, "ksSpecularEXP": 1.0}
    textures = {}
    out = mat.GetSurfaceOutput()
    src = out.GetConnectedSource() if out else None
    if src:
        sh = UsdShade.Shader(src[0].GetPrim())
        di = sh.GetInput("diffuseColor")
        if di:
            c = di.GetConnectedSource()
            if c:
                f = UsdShade.Shader(c[0].GetPrim()).GetInput("file")
                if f and f.Get():
                    ap = f.Get()
                    p = Path(ap.resolvedPath or ap.path)
                    textures["txDiffuse"] = p.name
                    copy_tex(p.name, resolved=p)
            elif di.Get() is not None:
                props["ksDiffuse"] = float(sum(di.Get()) / 3)
    return name, "ksPerPixel", props, textures, 0, False


def usd_to_kn5(doc, base_dir, keys, visual, physics, report):
    """Add every geometry.meshes USD file to the kn5 builders. Returns (n_visual, n_collision, has_collision)."""
    base_dir = Path(base_dir)
    nv = nc = 0
    used = set()

    def unique(n):
        n0, i = n, 1
        while n in used:
            i += 1
            n = f"{n0}_{i}"
        used.add(n)
        return n

    for entry in doc.get("geometry", {}).get("meshes", []):
        path = (base_dir / entry["uri"]).resolve()
        if path.suffix.lower() not in (".usd", ".usda", ".usdc", ".usdz") or not path.exists():
            report.add("unsupported", entry["uri"], "USD mesh file missing or not USD; skipped")
            continue
        stage = Usd.Stage.Open(str(path))
        roles = entry.get("roles", ["visual"])
        cache = UsdGeom.XformCache()

        def copy_tex(name, resolved=None, _dir=path.parent):
            for src in ([resolved] if resolved else []) + [_dir.parent / "textures" / name, _dir / name]:
                if src and Path(src).exists():
                    visual.texture(Path(src).name, Path(src).read_bytes())
                    return

        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            try:
                P, N, U, T = _mesh_arrays(UsdGeom.Mesh(prim), cache.GetLocalToWorldTransform(prim))
            except Exception as e:  # noqa: BLE001
                report.add("unsupported", str(prim.GetPath()), f"mesh not read: {e}")
                continue
            if len(T) == 0:
                continue
            collision = "collision" in roles and (roles == ["collision"] or bool(prim.GetAttribute("stj:collision").Get()))
            P, N = _to_ac(P), _to_ac(N)
            U = U.copy()
            U[:, 1] = 1.0 - U[:, 1]
            if collision:
                mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                m = re.match(r"STJ__surface__(.+)", mat.GetPrim().GetName()) if mat else None
                sid_ = m.group(1) if m else doc.get("surfaces", {}).get("default")
                key = keys.get(sid_, (sid_ or "ROAD").upper())
                pn = prim.GetName()
                base_name = pn if re.match(r"^\d{1,2}[A-Z_\-]", pn) else f"01{key}"
                bld, mi = physics, physics.material(sid_ or "surface", 0.3)
                nc += 1
            else:
                spec = _material_spec(prim, copy_tex)
                base_name = re.sub(r"[^A-Za-z0-9_\-]", "_", prim.GetName())
                bld = visual
                mi = visual.material_full(*spec) if spec else visual.material("default", 0.5)
                nv += 1
            step = LIMIT // 3
            for part, a in enumerate(range(0, len(T), step)):
                tri = T[a:a + step]
                uniq, inv = np.unique(tri.reshape(-1), return_inverse=True)
                nm = base_name if part == 0 else f"{base_name}_{part}"
                if collision and not re.search(r"\d$", nm):
                    nm += f"{part:03d}"
                fl = prim.GetCustomDataByKey("x_assetto_corsa") or {}
                bld.add_mesh(unique(nm), P[uniq], N[uniq], U[uniq], inv.reshape(-1), mi,
                             layer=int(fl.get("layer", 0)), visible=bool(fl.get("visible", not collision)) if not collision else False,
                             shadows=bool(fl.get("cast_shadows", False)), transparent=bool(fl.get("transparent", False)),
                             lod_in=float(fl.get("lod_in", 0.0)), lod_out=float(fl.get("lod_out", 0.0)))
    return nv, nc, nc > 0
