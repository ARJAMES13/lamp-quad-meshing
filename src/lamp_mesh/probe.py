"""Patch-layout probe: statistics of QuadWild's traced patch layout, before quantisation.

The field + trace stage writes, per surface triangle, the id of its patch (`m_rem_p0.patch`) and, per patch, its
number of corners (`m_rem_p0.corners`). Ten numbers summarise the layout relative to the element budget n*:
log #patches, #patches / n*, share of 4-corner patches, share with <= 3, >= 5 and >= 6 corners, coefficient of
variation of patch sizes, share of tiny patches (< 10 % of the median size), log(max/min patch size),
log #triangles. A failed trace gives all-NaN (which the value models learn to read as "QuadWild will fail").
"""
import os

import numpy as np

N_PROBE = 10


def probe_from_files(patch_path, corners_path, n_star):
    pid = np.loadtxt(patch_path, dtype=int, skiprows=1)
    cor = np.loadtxt(corners_path, dtype=int, skiprows=1).reshape(-1)
    sizes = np.bincount(pid)
    sizes = sizes[sizes > 0].astype(float)
    npch = len(cor)
    return [np.log(npch + 1), npch / n_star, np.mean(cor == 4), np.mean(cor <= 3), np.mean(cor >= 5), np.mean(cor >= 6),
            sizes.std() / (sizes.mean() + 1e-9), np.mean(sizes < 0.1 * np.median(sizes)), np.log(sizes.max() / (sizes.min() + 1)),
            np.log(len(pid))]


def probe_workdir(workdir, n_star):
    """Probe vector from a QuadWild working directory; None if the trace files are missing or unreadable."""
    try:
        return probe_from_files(os.path.join(workdir, "m_rem_p0.patch"), os.path.join(workdir, "m_rem_p0.corners"), n_star)
    except Exception:  # noqa: BLE001
        return None
