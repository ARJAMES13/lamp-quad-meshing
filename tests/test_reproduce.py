"""Regression test: the committed labels reproduce the reported LAMP numbers exactly."""
import numpy as np
import pytest

from _common import load_cfg, rel
from lamp_mesh import policy as P
from lamp_mesh.scoring import paired_bootstrap
from train import train


@pytest.fixture(scope="module")
def trained():
    cfg = load_cfg(rel("configs/lamp.toml"))
    models, lam, _, n = train(cfg, verbose=False)
    return models, lam, n, cfg["paths"], P.load_descriptors(rel(cfg["paths"]["descriptors"]))


def _eval(trained, nm):
    models, lam, _, paths, desc = trained
    X, S, T, Q, _ = P.table(rel(paths[f"{nm}_qw"]), rel(paths[f"{nm}_qqs"]), desc)
    v, t, _ = P.policy(models, X, S, T, lam)
    md, lo, hi = paired_bootstrap(v, Q, n=10000)
    return len(X), v.mean(), md, lo, hi, int((v < Q - .005).sum()), int((v > Q + .005).sum()), np.nanmean(t)


def test_cv_lambda(trained):
    assert trained[1] == 0.002 and trained[2] == 387


def test_test_headline(trained):
    n, mean, md, lo, hi, w, l_, t = _eval(trained, "test")
    assert (n, w, l_) == (141, 74, 6)
    assert [round(x, 4) for x in (mean, md, lo, hi)] == [0.3069, -0.0704, -0.0989, -0.0419] and round(t, 1) == 53.8


def test_ood_headline(trained):
    n, mean, md, lo, hi, w, l_, t = _eval(trained, "ood")
    assert (n, w, l_) == (40, 20, 1)
    assert [round(x, 4) for x in (mean, md, lo, hi)] == [0.2213, -0.0582, -0.0914, -0.0302] and round(t, 1) == 19.3
