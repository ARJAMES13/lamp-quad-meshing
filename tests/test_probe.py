import numpy as np

from lamp_mesh.policy import features
from lamp_mesh.probe import N_PROBE, probe_from_files, probe_workdir


def test_probe_statistics(tmp_path):
    pid = [0] * 50 + [1] * 30 + [2] * 2           # 3 patches, one tiny
    cor = [4, 4, 5]
    (tmp_path / "p.patch").write_text(f"{len(pid)}\n" + "\n".join(map(str, pid)) + "\n")
    (tmp_path / "p.corners").write_text(f"{len(cor)}\n" + "\n".join(map(str, cor)) + "\n")
    pr = probe_from_files(str(tmp_path / "p.patch"), str(tmp_path / "p.corners"), n_star=100)
    assert len(pr) == N_PROBE
    assert np.isclose(pr[1], 0.03) and np.isclose(pr[2], 2 / 3) and np.isclose(pr[4], 1 / 3)
    assert np.isclose(pr[7], 1 / 3)                # patch of size 2 is below 10 % of the median size (30)


def test_missing_trace_gives_nan_features(tmp_path):
    assert probe_workdir(str(tmp_path), 100) is None
    x = features(np.zeros(15), None, 2500, 10.0, 7)
    assert np.isnan(x[15:25]).all() and x[-1] == 1.0
