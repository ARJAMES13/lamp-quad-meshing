"""Quad mesh result container, OBJ I/O and Gmsh's own quad algorithms.

  run_gmsh_quads      Gmsh quad meshing: algo 11 = quasi-structured quads (QQS), algo 8 = Blossom recombination
All methods see the same healed CAD (geometry.import_cad).
"""
import time
from dataclasses import dataclass, field

import numpy as np

from .geometry import SurfaceSample, import_cad


@dataclass
class QuadResult:
    status: str                       # ok | extract_fail
    V: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    Q: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), int))
    T: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), int))   # leftover triangles
    time_s: float = 0.0
    other: int = 0                    # polygons with >4 sides
    msg: str = ""


def _read_poly_obj(path):
    vv, faces = [], []
    with open(path) as fh:
        for line in fh:
            t = line.split()
            if not t:
                continue
            if t[0] == "v":
                vv.append([float(x) for x in t[1:4]])
            elif t[0] == "f":
                faces.append([int(x.split("/")[0]) - 1 for x in t[1:]])
    return np.array(vv).reshape(-1, 3), faces


def _split_faces(faces):
    Q = np.array([f for f in faces if len(f) == 4], dtype=np.int64).reshape(-1, 4)
    T = np.array([f for f in faces if len(f) == 3], dtype=np.int64).reshape(-1, 3)
    return Q, T, sum(1 for f in faces if len(f) > 4)


def write_quad_obj(path, r: QuadResult):
    """Write a quad-dominant result as OBJ (quads and leftover triangles)."""
    with open(path, "w") as fh:
        fh.write("".join("v %.9g %.9g %.9g\n" % tuple(p) for p in r.V))
        fh.write("".join("f %d %d %d %d\n" % tuple(q + 1) for q in r.Q))
        fh.write("".join("f %d %d %d\n" % tuple(t + 1) for t in r.T))


def run_gmsh_quads(step_path: str, s: SurfaceSample, rel_edge: float, algo: int) -> QuadResult:
    import gmsh
    t0 = time.perf_counter()
    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        import_cad(step_path, 1.0, heal=True)
        h = rel_edge * s.diag
        for k, v in [("Mesh.MeshSizeMin", h), ("Mesh.MeshSizeMax", h), ("Mesh.MeshSizeFromCurvature", 0),
                     ("Mesh.Algorithm", algo), ("Mesh.RecombineAll", 1), ("Mesh.RecombinationAlgorithm", 1)]:
            gmsh.option.setNumber(k, v)
        gmsh.model.mesh.generate(2)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        V = xyz.reshape(-1, 3)
        tag2i = {int(t): i for i, t in enumerate(tags)}
        Q, T = [], []
        for et, _, en in zip(*gmsh.model.mesh.getElements(2)):
            if et == 3:
                Q.append(np.vectorize(tag2i.get)(en.reshape(-1, 4)))
            elif et == 2:
                T.append(np.vectorize(tag2i.get)(en.reshape(-1, 3)))
    finally:
        gmsh.finalize()
    Q = np.concatenate(Q) if Q else np.zeros((0, 4), np.int64)
    T = np.concatenate(T) if T else np.zeros((0, 3), np.int64)
    used = np.unique(np.r_[Q.ravel(), T.ravel()]).astype(np.int64)
    remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
    return QuadResult("ok", V[used], remap[Q], remap[T], time.perf_counter() - t0)
