"""QuadWild + Bi-MDF quantisation (Pietroni et al. 2021; Heistermann et al. 2023) through the QRemeshify bindings.

Input: the CAD reference triangulation + the exact CAD sharp edges, written as QuadWild '.sharp' features.
The field + trace stage runs once (`QuadWild(...)`); quantisation is re-run with a scale factor to hit a budget.
The QRemeshify package directory (the folder that contains `lib/`) is taken from the `QREMESHIFY_DIR` environment
variable, or from `third_party/qremeshify/QRemeshify` in the repository.
"""
import os
import sys
import time

import numpy as np

_DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "third_party", "qremeshify", "QRemeshify")


def qremeshify_dir():
    return os.environ.get("QREMESHIFY_DIR", _DEFAULT_DIR)


def _lib():
    d = qremeshify_dir()
    if not os.path.isdir(os.path.join(d, "lib")):
        raise RuntimeError(f"QRemeshify not found at {d}; see third_party/qremeshify/README.md")
    if d not in sys.path:
        sys.path.insert(0, d)
    import lib as qwlib  # QRemeshify/lib/__init__.py (ctypes bindings)
    return qwlib


def write_inputs(V, F, feat_edges, mesh_path, sharp_path):
    with open(mesh_path, "w", encoding="utf-8") as f:
        f.write("".join(f"v {x:.9g} {y:.9g} {z:.9g}\n" for x, y, z in V))
        f.write("".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in F))
    fe = {tuple(sorted(e)) for e in np.asarray(feat_edges).tolist()}
    rows = []
    for fi, tri in enumerate(F):
        for k in range(3):
            e = tuple(sorted((int(tri[k]), int(tri[(k + 1) % 3]))))
            if e in fe:
                rows.append(f"0,{fi},{k}")
                fe.discard(e)                                  # each edge once
    with open(sharp_path, "w", encoding="utf-8") as f:
        f.write(f"{len(rows)}\n" + "".join(r + "\n" for r in rows))
    return len(rows)


def read_poly_obj(path):
    V, Q, T = [], [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                (Q if len(idx) == 4 else T if len(idx) == 3 else Q).append(idx[:4] if len(idx) >= 4 else idx)
    return np.array(V), np.array([q for q in Q if len(q) == 4], np.int64).reshape(-1, 4), np.array(T, np.int64).reshape(-1, 3)


class QuadWild:
    def __init__(self, V, F, feat_edges, workdir, remesh=True, sharp_angle=35.0):
        qw = _lib()
        os.makedirs(workdir, exist_ok=True)
        self.workdir = workdir
        self.mesh_path = os.path.join(workdir, "m.obj")
        self.q = qw.Quadwild(self.mesh_path)
        self.n_sharp = write_inputs(V, F, feat_edges, self.mesh_path, self.q.sharp_path)
        t0 = time.perf_counter()
        self.q.remeshAndField(remesh=remesh, enableSharp=True, sharpAngle=sharp_angle)
        self.q.trace()
        self.t_prep = time.perf_counter() - t0

    def quadrangulate(self, scale=1.0, smooth=True, time_limit=60, alpha=0.005, flow="SIMPLE"):
        t0 = time.perf_counter()
        self.q.quadrangulate(smooth, scale, 0, alpha, "LEASTSQUARES", time_limit, 0.0, 0.4, True, True, True, 0.9,
                             True, 0.1, True, False, False, True, True, flow, "DEFAULT",
                             [3.0, 5.0, 10.0, 20.0, 30.0, 60.0, 90.0, 120.0], [0.005, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3])
        out = self.q.output_smoothed_path if smooth and os.path.exists(self.q.output_smoothed_path) else self.q.output_path
        V, Q, T = read_poly_obj(out)
        return V, Q, T, time.perf_counter() - t0


def calibrated(qw, n_star, tol=0.03, max_runs=5):
    """Re-run quantisation (up to max_runs) rescaling by sqrt(n / n_star) until within tol in log(n).
    Returns (V, Q, T, n, t_last, t_all, history); V is None if every run failed."""
    scale, best, t_all, hist = 1.0, None, 0.0, []
    for _ in range(max_runs):
        V, Q, T, t = qw.quadrangulate(scale)
        t_all += t
        n = len(Q) + len(T)
        hist.append((round(scale, 4), n))
        if n and (best is None or abs(np.log(n / n_star)) < abs(np.log(best[3] / n_star))):
            best = (V, Q, T, n, t)
        if n == 0 or abs(np.log(n / n_star)) < tol:
            break
        scale *= (n / n_star) ** 0.5
    if best is None:
        return None, None, None, 0, 0.0, t_all, hist
    return (*best, t_all, hist)
