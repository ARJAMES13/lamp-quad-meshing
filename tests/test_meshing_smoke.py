"""Smoke tests for the CAD -> surface sample -> quad mesh -> metrics pipeline on tiny generated parts."""
import numpy as np
import pytest

from lamp_mesh.geometry import build_surface
from lamp_mesh.meshers import run_gmsh_quads
from lamp_mesh.metrics import quad_metrics
from lamp_mesh.qqs import run_qqs
from lamp_mesh.scoring import composite_score


@pytest.fixture(scope="module")
def plate(shapes):
    return shapes["plate"], build_surface(shapes["plate"], rel_h=0.03)


def test_surface_sample(plate):
    _, s = plate
    assert len(s.F) > 100 and len(s.feat_edges) > 0 and np.isfinite(s.attrs).all()


def test_blossom_quads_and_metrics(plate):
    path, s = plate
    m = quad_metrics(run_gmsh_quads(path, s, 0.05, 8), s)
    assert m["valid"] and m["n_quads"] > 50 and 0 <= composite_score(m) < 10


def test_qqs_budget(plate):
    path, s = plate
    r = run_qqs(path, s, None, n_target=400)
    assert len(r.Q) > 0 and 0.5 < len(r.Q) / 400 < 2.0


def test_quadwild_probe_and_budget(plate, tmp_path):
    from lamp_mesh.meshers import QuadResult
    from lamp_mesh.probe import N_PROBE, probe_workdir
    from lamp_mesh.quadwild import QuadWild, calibrated, qremeshify_dir
    import os
    if not os.path.isdir(os.path.join(qremeshify_dir(), "lib")):
        pytest.skip("QRemeshify not installed (third_party/qremeshify/README.md)")
    path, s = plate
    qw = QuadWild(s.V, s.F, s.feat_edges, str(tmp_path / "qw"))
    pr = probe_workdir(str(tmp_path / "qw"), 400)
    assert pr is not None and len(pr) == N_PROBE and np.isfinite(pr).all()
    V, Q, T, n, t, _, _ = calibrated(qw, 400)
    assert V is not None and 0.7 < n / 400 < 1.4
    assert quad_metrics(QuadResult("ok", V, Q, T, t), s)["valid"]
