"""Minimal KN5 reader: textures and meshes are skipped, only the node tree (names + transforms) is kept.

That is what a track converter needs: AC_START_*, AC_PIT_*, AC_TIME_*_L/R, AC_HOTLAP_START_* empties
and the names/bounds of mesh nodes (1ROAD, 1KERB, ...).
"""
import struct


class Node:
    __slots__ = ("name", "kind", "matrix", "children", "mesh")

    def __init__(self, name, kind):
        self.name, self.kind, self.matrix, self.children, self.mesh = name, kind, None, [], None

    def walk(self, parent=None, parent_world=None):
        """Yield (node, world_matrix) where matrices are 4x4 row-major tuples (DirectX row-vector convention)."""
        world = mat_mul(self.matrix, parent_world) if (self.matrix and parent_world) else (self.matrix or parent_world)
        yield self, world
        for c in self.children:
            yield from c.walk(self, world)


def mat_mul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)) for i in range(4))


class _R:
    def __init__(self, data):
        self.d, self.p = data, 0

    def unpack(self, fmt):
        v = struct.unpack_from("<" + fmt, self.d, self.p)
        self.p += struct.calcsize("<" + fmt)
        return v if len(v) > 1 else v[0]

    def skip(self, n):
        self.p += n

    def string(self):
        n = self.unpack("i")
        s = self.d[self.p:self.p + n].decode("utf-8", "replace")
        self.p += n
        return s


class Kn5Data:
    """Everything a kn5 holds beyond the node tree: materials and texture blobs (offsets, not copies)."""

    def __init__(self):
        self.materials = []   # dicts: name, shader, alpha_blend, alpha_tested, props{name: value}, textures{sampler: name}
        self.textures = {}    # name -> (offset, size) into .raw
        self.raw = b""

    def texture_bytes(self, name):
        o, n = self.textures[name]
        return self.raw[o:o + n]


def read_kn5(path, keep_geometry=False, data=None):
    """Return (version, root Node). Meshes keep vertex/index counts, bounding sphere and material id;
    with keep_geometry also numpy arrays (positions, normals, uvs in AC frame, triangle indices).
    Pass a Kn5Data to receive materials and texture offsets."""
    r = _R(open(path, "rb").read())
    if data is not None:
        data.raw = r.d
    if r.d[:6] != b"sc6969":
        raise ValueError("not a KN5 file: " + str(path))
    r.skip(6)
    version = r.unpack("i")
    if version > 5:
        r.unpack("i")
    # textures
    for _ in range(r.unpack("i")):
        r.unpack("i")
        tname = r.string()
        size = r.unpack("i")
        if data is not None:
            data.textures[tname] = (r.p, size)
        r.skip(size)
    # materials
    materials = []
    for _ in range(r.unpack("i")):
        name, shader = r.string(), r.string()
        alpha_blend, alpha_tested = r.unpack("BB")
        if version > 4:
            r.unpack("i")
        props, texs = {}, {}
        for _ in range(r.unpack("i")):
            pn = r.string()
            props[pn] = r.unpack("f")
            r.skip(36)
        for _ in range(r.unpack("i")):
            sampler = r.string()
            r.skip(4)
            texs[sampler] = r.string()
        materials.append(name)
        if data is not None:
            data.materials.append({"name": name, "shader": shader, "alpha_blend": alpha_blend,
                                   "alpha_tested": bool(alpha_tested), "props": props, "textures": texs})
    root = _node(r, version, materials, keep_geometry)
    return version, root


def _node(r, version, materials, keep=False):
    kind = r.unpack("i")
    name = r.string()
    nchild = r.unpack("i")
    r.unpack("?")
    node = Node(name, {1: "dummy", 2: "mesh", 3: "skinned"}.get(kind, str(kind)))
    if kind == 1:
        f = r.unpack("16f")
        node.matrix = tuple(tuple(f[i * 4:i * 4 + 4]) for i in range(4))
    elif kind in (2, 3):
        flags = r.unpack("3?" if version > 4 else "2?")  # cast shadows, visible, transparent
        if kind == 3:
            nb = r.unpack("i")
            for _ in range(nb):
                r.string()
                r.skip(64)
            nv = r.unpack("i")
            r.skip(nv * 76)
        else:
            nv = r.unpack("i")
            vb = None
            if keep:
                import numpy as np
                vb = np.frombuffer(r.d, dtype="<f4", count=nv * 11, offset=r.p).reshape(nv, 11)
            r.skip(nv * 44)
        ni = r.unpack("i")
        ib = None
        if keep and kind == 2:
            import numpy as np
            ib = np.frombuffer(r.d, dtype="<u2", count=ni, offset=r.p)
        r.skip(ni * 2)
        mat = r.unpack("i")
        layer = r.unpack("i")
        lod_in, lod_out = r.unpack("ff")
        bs = r.unpack("4f")
        r.unpack("?")  # renderable
        if kind == 3:
            pass
        node.matrix = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
        node.mesh = {"vertices": nv, "indices": ni, "material": materials[mat] if 0 <= mat < len(materials) else None,
                     "sphere": bs,
                     "cast_shadows": bool(flags[0]), "visible": bool(flags[1]),
                     "transparent": bool(flags[2]) if len(flags) > 2 else False,
                     "layer": layer, "lod_in": lod_in, "lod_out": lod_out}
        if keep and kind == 2:
            node.mesh.update({"positions": vb[:, 0:3], "normals": vb[:, 3:6], "uvs": vb[:, 6:8], "triangles": ib.reshape(-1, 3)})
    for _ in range(nchild):
        node.children.append(_node(r, version, materials, keep))
    return node


def world_position(world):
    """Translation of a row-vector matrix, in AC coordinates (x right, y up, z forward)."""
    return world[3][0], world[3][1], world[3][2]


def world_forward_xz(world):
    """Forward (+z) axis of the node projected on the AC ground plane (x, z)."""
    return world[2][0], world[2][2]
