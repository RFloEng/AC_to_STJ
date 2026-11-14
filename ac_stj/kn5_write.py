"""Minimal KN5 writer (version 5, no textures): flat-shaded meshes plus named dummy nodes.

Enough for a drivable road ribbon, grass margins, and the AC_START / AC_PIT / AC_TIME / AC_HOTLAP_START empties.
Only what AC needs to load a track; no UV unwrapping, no lighting, no textures.
"""
import struct

import numpy as np

IDENT = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))


def _s(text):
    b = text.encode("utf-8")
    return struct.pack("<i", len(b)) + b


class Kn5Builder:
    def __init__(self, root_name):
        self.root_name = root_name
        self.materials = []   # (name, spec dict)
        self.children = []    # serialized node blobs
        self.textures = {}    # name -> bytes

    def material(self, name, diffuse=0.5, ambient=0.4, specular=0.0):
        props = {"ksAmbient": ambient, "ksDiffuse": diffuse, "ksSpecular": specular, "ksSpecularEXP": 1.0}
        return self.material_full(name, "ksPerPixel", props)

    def material_full(self, name, shader, props, textures=None, alpha_blend=0, alpha_tested=False):
        for i, (n, _) in enumerate(self.materials):
            if n == name:
                return i
        self.materials.append((name, {"shader": shader, "props": dict(props), "textures": dict(textures or {}),
                                      "alpha_blend": int(alpha_blend), "alpha_tested": bool(alpha_tested)}))
        return len(self.materials) - 1

    def texture(self, name, data):
        self.textures[name] = data

    def add_dummy(self, name, matrix):
        flat = [v for row in matrix for v in row]
        self.children.append(struct.pack("<i", 1) + _s(name) + struct.pack("<i", 0) + b"\x01" + struct.pack("<16f", *flat))

    def add_mesh(self, name, verts, normals, uvs, indices, material, layer=0, visible=True, shadows=False,
                 transparent=False, lod_in=0.0, lod_out=0.0):
        """verts/normals (n,3), uvs (n,2), indices (m,) uint16 triangle list, in AC coordinates."""
        v = np.asarray(verts, dtype="<f4")
        nrm = np.asarray(normals, dtype="<f4")
        uv = np.asarray(uvs, dtype="<f4")
        tan = np.zeros_like(v)
        tan[:, 0] = 1.0
        vb = np.concatenate([v, nrm, uv, tan], axis=1).astype("<f4")
        idx = np.asarray(indices, dtype="<u2")
        centre = v.mean(axis=0)
        rad = float(np.linalg.norm(v - centre, axis=1).max()) if len(v) else 0.0
        blob = struct.pack("<i", 2) + _s(name) + struct.pack("<i", 0) + b"\x01"
        blob += struct.pack("<???", shadows, visible, transparent)
        blob += struct.pack("<i", len(v)) + vb.tobytes()
        blob += struct.pack("<i", len(idx)) + idx.tobytes()
        blob += struct.pack("<ii", material, layer) + struct.pack("<ff", lod_in, lod_out)
        blob += struct.pack("<4f", centre[0], centre[1], centre[2], rad) + b"\x01"
        self.children.append(blob)

    def write(self, path):
        out = [b"sc6969", struct.pack("<i", 5), struct.pack("<i", len(self.textures))]
        for tname, blob in self.textures.items():
            out.append(struct.pack("<i", 1) + _s(tname) + struct.pack("<i", len(blob)) + blob)
        out.append(struct.pack("<i", len(self.materials)))
        for name, m in self.materials:
            out.append(_s(name) + _s(m["shader"]) + struct.pack("<BB", m["alpha_blend"], m["alpha_tested"])
                       + struct.pack("<i", 0))
            out.append(struct.pack("<i", len(m["props"])))
            for pn, pv in m["props"].items():
                out.append(_s(pn) + struct.pack("<f", pv) + b"\x00" * 36)
            out.append(struct.pack("<i", len(m["textures"])))
            for slot, (sampler, tn) in enumerate(m["textures"].items()):
                out.append(_s(sampler) + struct.pack("<i", slot) + _s(tn))
        out.append(struct.pack("<i", 1) + _s(self.root_name) + struct.pack("<i", len(self.children)) + b"\x01")
        out.append(struct.pack("<16f", *[v for row in IDENT for v in row]))
        out.extend(self.children)
        with open(path, "wb") as f:
            f.write(b"".join(out))


def node_matrix(position_ac, forward_xz):
    """Row-vector matrix with +z forward, +y up, +x right (AC left-handed frame)."""
    fx, fz = forward_xz
    n = (fx * fx + fz * fz) ** 0.5 or 1.0
    fx, fz = fx / n, fz / n
    px, py, pz = position_ac
    return ((fz, 0.0, -fx, 0.0), (0.0, 1.0, 0.0, 0.0), (fx, 0.0, fz, 0.0), (px, py, pz, 1.0))
