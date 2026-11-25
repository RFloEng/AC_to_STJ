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
    props = {"ksAmbient": 0.35, "ksDiffuse": 0.35, "ksSpecular": 0.0, "ksSpecularEXP": 1.0}   # median of stock AC tracks (0.35 / 0.33)
    textures = {}
    out = mat.GetSurfaceOutput()
    src = out.GetConnectedSource() if out else None
    if src:
        sh = UsdShade.Shader(src[0].GetPrim())
        di = sh.GetInput("diffuseColor")
        oi = sh.GetInput("opacity")
        ot = sh.GetInput("opacityThreshold")
        cutout = bool(oi and oi.GetConnectedSource())          # opacity from a texture: leaves, fences, grass cards
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
    if cutout and "txDiffuse" in textures:
        if ot is not None and ot.Get():
            props["ksAlphaRef"] = float(ot.Get())
            return name, "ksPerPixelAT", props, textures, 0, True
        return name, "ksPerPixelAlpha", props, textures, 1, False
    return name, "ksPerPixel", props, textures, 0, False


def _tri_faces(mesh):
    """Face index of every triangle _mesh_arrays returns (same order)."""
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    return np.repeat(np.arange(len(counts)), np.maximum(counts - 2, 0))


def _take(P, N, U, T, sel):
    """Sub-mesh made of triangles T[sel], vertices compacted."""
    tri = T[sel]
    uniq, inv = np.unique(tri.reshape(-1), return_inverse=True)
    return P[uniq], N[uniq], U[uniq], inv.reshape(-1, 3)


def _ground_spec(name, doc, copy_tex, base_dir, which=0):
    """Ground material (STJ ground.materials[] + ground.layers) -> kn5 material using its first layer's albedo.
    AC has no per-vertex layer blending, so the palette's first layer stands for the whole material; the texture repeat of
    DR2's Map_UVScaleAndOffset goes into `uv` (u, v multipliers) for the caller to apply."""
    g = doc.get("ground") or {}
    mat = next((m for m in g.get("materials", []) if m.get("material") == name), None)
    if not mat or not mat.get("palette"):
        return None
    which = min(which, len(mat["palette"]) - 1)
    layer = (g.get("layers") or {}).get(mat["palette"][which])
    if not layer or "albedo" not in layer:
        return None
    rel = layer["albedo"]["uri"]
    copy_tex(Path(rel).name, resolved=Path(base_dir) / rel)
    sc = (layer.get("x_dr2") or {}).get("Map_UVScaleAndOffset") or [1.0, 1.0]
    props = {"ksAmbient": 0.35, "ksDiffuse": 0.35, "ksSpecular": 0.0, "ksSpecularEXP": 1.0}
    return (f"{name}_ground{which}", "ksPerPixel", props, {"txDiffuse": Path(rel).name}, 0, False), (float(sc[0]), float(sc[1]))


def _tri_corners(mesh):
    """Corner indices (into face-varying arrays) of every triangle, same order as _mesh_arrays."""
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    if np.all(counts == 3):
        return starts[:, None] + np.arange(3)[None, :]
    return np.array([(st, st + k, st + k + 1) for st, c in zip(starts, counts) for k in range(1, c - 1)],
                    dtype=np.int64).reshape(-1, 3)


def _top_layer(mesh, which_tris):
    """For a two-layer ground blend: 1 where the second palette layer dominates the triangle (mean of its vertex weight > 0.5)."""
    pv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("stj_surf_weights")
    if not pv or pv.Get() is None:
        return None
    w = np.array(pv.Get(), dtype=np.float64).reshape(len(pv.Get()), -1)
    if w.shape[1] < 2:
        return None
    corners = _tri_corners(mesh)[which_tris]
    if pv.GetInterpolation() == "vertex":
        idx = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        corners = idx[corners]
    return (w[corners, 1].mean(axis=1) > 0.5).astype(np.int64)


def _material_groups(prim, ftri, doc, copy_tex, base_dir):
    """[(spec, triangle indices or None = all, (u_scale, v_scale) or None)] for a mesh, honouring GeomSubset material bindings."""
    mesh = UsdGeom.Mesh(prim)
    subsets = UsdGeom.Subset.GetAllGeomSubsets(mesh)

    def resolve(p):
        spec = _material_spec(p, copy_tex)
        mat, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()
        name = mat.GetPrim().GetName() if mat else None
        if name and (spec is None or "txDiffuse" not in spec[3]):
            g = _ground_spec(name, doc, copy_tex, base_dir)
            if g:
                return g[0], g[1]
        return spec, None

    if not subsets:
        spec, uv = resolve(prim)
        return [(spec, None, uv)]
    out, covered = [], np.zeros(int(ftri.max()) + 1 if len(ftri) else 0, dtype=bool)
    for sub in subsets:
        faces = np.array(sub.GetIndicesAttr().Get() or [], dtype=np.int64)
        if len(faces) == 0:
            continue
        covered[faces[faces < len(covered)]] = True
        sel = np.nonzero(np.isin(ftri, faces))[0]
        spec, uv = resolve(sub.GetPrim())
        if uv is not None and spec and spec[0].endswith("_ground0"):
            top = _top_layer(mesh, sel)
            if top is not None and top.any():
                mat, _ = UsdShade.MaterialBindingAPI(sub.GetPrim()).ComputeBoundMaterial()
                g1 = _ground_spec(mat.GetPrim().GetName(), doc, copy_tex, base_dir, 1)
                if g1:
                    out.append((spec, sel[top == 0], uv))
                    out.append((g1[0], sel[top == 1], g1[1]))
                    continue
        out.append((spec, sel, uv))
    rest = np.nonzero(~covered[ftri])[0] if len(ftri) else np.array([], dtype=np.int64)
    if len(rest):
        spec, uv = resolve(prim)
        out.append((spec, rest, uv))
    return out


def _two_sided(mesh, P, N, U, T):
    """doubleSided meshes: AC culls back faces, so add a reversed copy of the triangles (normals flipped) to show both sides."""
    attr = mesh.GetDoubleSidedAttr()
    if not (attr and attr.Get()):
        return P, N, U, T
    n = len(P)
    return (np.concatenate([P, P]), np.concatenate([N, -N]), np.concatenate([U, U]),
            np.concatenate([T, T[:, ::-1] + n]))


def _instance_transforms(inst):
    """PointInstancer -> (protoIndices, 3x3 row-vector matrices scale*rotation (n,3,3), positions (n,3))."""
    pos = np.array(inst.GetPositionsAttr().Get() or [], dtype=np.float64).reshape(-1, 3)
    n = len(pos)
    idx = np.array(inst.GetProtoIndicesAttr().Get() or [], dtype=np.int64)
    ori = inst.GetOrientationsAttr().Get()
    sc = inst.GetScalesAttr().Get()
    S = np.array(sc, dtype=np.float64).reshape(-1, 3) if sc is not None and len(sc) else np.ones((n, 3))
    R = np.tile(np.eye(3), (n, 1, 1))
    if ori is not None and n > 0 and len(ori) == n:
        q = np.array([[o.GetReal(), *o.GetImaginary()] for o in ori], dtype=np.float64)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        w, x, y, z = q.T
        Rc = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                       2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                       2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1).reshape(-1, 3, 3)
        R = Rc.transpose(0, 2, 1)                                # row-vector convention
    return idx, S[:, :, None] * R, pos


_MAT_CACHE = {}


def _line_dist(doc):
    """Function pos(n,3 STJ frame) -> plan-view distance (m) to the reference line (sampled every ~10 m)."""
    pts = np.array(doc["reference"]["centerline"]["points"], dtype=np.float64)[:, :2]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    ch = np.concatenate([[0], np.cumsum(seg)])
    line = pts[np.unique(np.searchsorted(ch, np.arange(0, ch[-1] + 10.0, 10.0)).clip(0, len(pts) - 1))]

    def dist(p):
        out = np.empty(len(p))
        for a in range(0, len(p), 20000):
            q = p[a:a + 20000, :2]
            out[a:a + 20000] = np.sqrt(((q[:, None, :] - line[None, :, :]) ** 2).sum(axis=2).min(axis=1))
        return out
    return dist


def _proto_cost(inst, stage):
    """Vertices one placement of each prototype adds to the kn5 (double-sided meshes are doubled)."""
    cost = []
    for t in inst.GetPrototypesRel().GetTargets():
        v = 0
        for m in Usd.PrimRange(stage.GetPrimAtPath(t)):
            if m.IsA(UsdGeom.Mesh):
                mm = UsdGeom.Mesh(m)
                v += len(mm.GetPointsAttr().Get() or []) * (2 if mm.GetDoubleSidedAttr().Get() else 1)
        cost.append(v)
    return np.array(cost, dtype=np.float64)


def _pick_radius(stage, cache, distf, radius, budget):
    """Distance within which instanced vertices fit `budget` (0 = no budget); never more than `radius` (0 = no limit)."""
    ds, cs = [], []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.PointInstancer):
            continue
        inst = UsdGeom.PointInstancer(prim)
        idx, _, pos = _instance_transforms(inst)
        if len(idx) == 0:
            continue
        W = np.array(cache.GetLocalToWorldTransform(prim)).reshape(4, 4)
        ds.append(distf(pos @ W[:3, :3] + W[3, :3]))
        cs.append(_proto_cost(inst, stage)[idx])
    if not ds:
        return radius
    d, c = np.concatenate(ds), np.concatenate(cs)
    r = radius if radius and radius > 0 else float("inf")
    if budget and budget > 0 and c[d <= r].sum() > budget:
        order = np.argsort(d)
        csum = np.cumsum(c[order])
        r = min(r, float(d[order][np.searchsorted(csum, budget).clip(0, len(d) - 1)]))
    return r


def _emit_instances(instancer, cache, roles, doc, keys, visual, physics, report, used, unique, copy_tex, distf=None, r_eff=None, stats=None, base_dir=None):
    """Bake a PointInstancer into ordinary kn5 meshes: every prototype mesh, copied to all its placements (visual only)."""
    if "visual" not in roles:
        return 0
    inst = UsdGeom.PointInstancer(instancer)
    idx, SR, pos = _instance_transforms(inst)
    if len(idx) == 0:
        return 0
    W = np.array(cache.GetLocalToWorldTransform(instancer)).reshape(4, 4)
    if distf is not None and r_eff is not None and np.isfinite(r_eff):
        keep = distf(pos @ W[:3, :3] + W[3, :3]) <= r_eff
        if stats is not None:
            stats["placed"] += int(keep.sum())
            stats["culled"] += int((~keep).sum())
        idx = np.where(keep, idx, -1)
        if not keep.any():
            return 0
    Winv = np.linalg.inv(W)
    lod_out = instancer.GetAttribute("stj:lodOut").Get() if instancer.GetAttribute("stj:lodOut") else None
    lod_in = instancer.GetAttribute("stj:lodIn").Get() if instancer.GetAttribute("stj:lodIn") else None
    count = 0
    stage = instancer.GetStage()
    base = re.sub(r"[^A-Za-z0-9_\-]", "_", instancer.GetParent().GetName() or instancer.GetName())
    for pi, target in enumerate(inst.GetPrototypesRel().GetTargets()):
        who = np.nonzero(idx == pi)[0]
        if len(who) == 0:
            continue
        root = stage.GetPrimAtPath(target)
        for mprim in Usd.PrimRange(root):
            if not mprim.IsA(UsdGeom.Mesh):
                continue
            m_inst = np.array(cache.GetLocalToWorldTransform(mprim)).reshape(4, 4) @ Winv
            try:
                P, N, U, T = _mesh_arrays(UsdGeom.Mesh(mprim), m_inst)
            except Exception as e:  # noqa: BLE001
                report.add("unsupported", str(mprim.GetPath()), f"instanced mesh not read: {e}")
                continue
            if len(T) == 0:
                continue
            # _mesh_arrays applied m_inst to the points only; normals rotate with it too
            N = N @ m_inst[:3, :3]
            for gi, (spec, sel, uvs) in enumerate(_material_groups(mprim, _tri_faces(UsdGeom.Mesh(mprim)), doc, copy_tex, base_dir)):
                if sel is not None:
                    Pg, Ng, Ug, Tg = _take(P, N, U, T, sel)
                else:
                    Pg, Ng, Ug, Tg = P, N, U, T
                if len(Tg) == 0:
                    continue
                if uvs:
                    Ug = Ug * np.array(uvs)
                Pg, Ng, Ug, Tg = _two_sided(UsdGeom.Mesh(mprim), Pg, Ng, Ug, Tg)
                key = (spec[0], spec[1]) if spec else None
                if key not in _MAT_CACHE or _MAT_CACHE[key][0] is not visual:
                    _MAT_CACHE[key] = (visual, visual.material_full(*spec) if spec else visual.material("default", 0.5))
                mi = _MAT_CACHE[key][1]
                per = max(1, LIMIT // max(len(Pg), 1))
                for c0 in range(0, len(who), per):
                    ids = who[c0:c0 + per]
                    Pl = np.einsum("vj,ijk->ivk", Pg, SR[ids]) + pos[ids][:, None, :]          # (n, v, 3) in instancer space
                    Pw = Pl @ W[:3, :3] + W[3, :3]
                    Nl = np.einsum("vj,ijk->ivk", Ng, SR[ids]) @ W[:3, :3]
                    Nl /= np.where(np.linalg.norm(Nl, axis=2, keepdims=True) > 0, np.linalg.norm(Nl, axis=2, keepdims=True), 1)
                    n, v = len(ids), len(Pg)
                    allP, allN = _to_ac(Pw.reshape(-1, 3)), _to_ac(Nl.reshape(-1, 3))
                    allU = np.tile(Ug, (n, 1)).copy()
                    allU[:, 1] = 1.0 - allU[:, 1]
                    allT = (Tg[None, :, :] + (np.arange(n) * v)[:, None, None]).reshape(-1, 3)
                    nm = unique(f"{base}_{mprim.GetName()}" + (f"_g{gi}" if gi else "") + (f"_{c0 // per}" if c0 else ""))
                    visual.add_mesh(nm, allP, allN, allU, allT.reshape(-1), mi, visible=True, shadows=False,
                                    lod_in=float(lod_in or 0.0), lod_out=float(lod_out or 0.0))
                    count += 1
    return count


def usd_to_kn5(doc, base_dir, keys, visual, physics, report, instance_radius=0.0, instance_budget=12e6):
    """Add every geometry.meshes USD file to the kn5 builders. Returns (n_visual, n_collision, has_collision)."""
    base_dir = Path(base_dir)
    nv = nc = 0
    used = set()
    distf = _line_dist(doc)
    stats = {"placed": 0, "culled": 0}

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

        r_eff = _pick_radius(stage, cache, distf, instance_radius, instance_budget)
        proto_roots = set()
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.PointInstancer):
                proto_roots.update(str(t) for t in UsdGeom.PointInstancer(prim).GetPrototypesRel().GetTargets())

        def under_proto(prim):
            sp = str(prim.GetPath())
            return any(sp == r or sp.startswith(r + "/") for r in proto_roots)

        work = []                                   # (prim, xf, instancer or None)
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.PointInstancer):
                work.append((prim, None, prim))
            elif prim.IsA(UsdGeom.Mesh) and not under_proto(prim):
                work.append((prim, cache.GetLocalToWorldTransform(prim), None))
        for prim, xf, instancer in work:
            if instancer is not None:
                nv += _emit_instances(instancer, cache, roles, doc, keys, visual, physics, report, used, unique, copy_tex, distf, r_eff, stats, base_dir)
                continue
            try:
                P, N, U, T = _mesh_arrays(UsdGeom.Mesh(prim), xf)
            except Exception as e:  # noqa: BLE001
                report.add("unsupported", str(prim.GetPath()), f"mesh not read: {e}")
                continue
            if len(T) == 0:
                continue
            collision = "collision" in roles and (roles == ["collision"] or bool(prim.GetAttribute("stj:collision").Get()))
            # decals stay exactly where the source game has them (no lift); they are flagged transparent and put on a later
            # layer so AC draws them after the terrain. This is a draw-order hint, not a depth bias: flicker is possible.
            is_decal = "decal" in roles and not collision
            if collision:
                groups = [(None, None, None)]
            else:
                groups = _material_groups(prim, _tri_faces(UsdGeom.Mesh(prim)), doc, copy_tex, base_dir)
            for gi, (spec, sel, uvs) in enumerate(groups):
                if sel is not None:
                    Pg, Ng, Ug, Tg = _take(P, N, U, T, sel)
                else:
                    Pg, Ng, Ug, Tg = P, N, U, T
                if len(Tg) == 0:
                    continue
                if uvs:
                    Ug = Ug * np.array(uvs)
                if not collision:
                    Pg, Ng, Ug, Tg = _two_sided(UsdGeom.Mesh(prim), Pg, Ng, Ug, Tg)
                Pg, Ng = _to_ac(Pg), _to_ac(Ng)
                Ug = Ug.copy()
                Ug[:, 1] = 1.0 - Ug[:, 1]
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
                    base_name = re.sub(r"[^A-Za-z0-9_\-]", "_", prim.GetName()) + (f"_g{gi}" if gi else "")
                    bld = visual
                    mi = visual.material_full(*spec) if spec else visual.material("default", 0.5)
                    nv += 1
                step = LIMIT // 3
                for part, a in enumerate(range(0, len(Tg), step)):
                    tri = Tg[a:a + step]
                    uniq, inv = np.unique(tri.reshape(-1), return_inverse=True)
                    nm = base_name if part == 0 else f"{base_name}_{part}"
                    if collision and not re.search(r"\d$", nm):
                        nm += f"{part:03d}"
                    fl = prim.GetCustomDataByKey("x_assetto_corsa") or {}
                    bld.add_mesh(unique(nm), Pg[uniq], Ng[uniq], Ug[uniq], inv.reshape(-1), mi,
                                 layer=int(fl.get("layer", 0)) + (1 if is_decal else 0),
                                 visible=bool(fl.get("visible", not collision)) if not collision else False,
                                 shadows=bool(fl.get("cast_shadows", False)) and not is_decal,
                                 transparent=bool(fl.get("transparent", False)) or is_decal,
                                 lod_in=float(fl.get("lod_in", 0.0)), lod_out=float(fl.get("lod_out", 0.0)))
    if stats["placed"] or stats["culled"]:
        report.add("approximated", "geometry.meshes instancers",
                   f"AC has no instancing: {stats['placed']} placements baked into meshes, {stats['culled']} left out because they are "
                   f"farther from the reference line than the vertex budget allows (budget {instance_budget / 1e6:g} M instanced vertices, "
                   f"radius limit {instance_radius:g} m; --instance-budget 0 --instance-radius 0 keeps all)")
    return nv, nc, nc > 0
