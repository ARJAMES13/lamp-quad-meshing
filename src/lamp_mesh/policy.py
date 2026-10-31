"""LAMP: learned controller over the QuadWild (QW) kernel with Gmsh QQS as the alternative.

Probe = QuadWild's field + trace stage (patch layout, before quantisation). Features = 15-D geometry descriptor
+ 10 probe statistics + log budget + probe time + #sharp edges + trace-failed flag. Actions after the probe:
finish QW | run QQS | both (keep the better a-priori score). One gradient-boosted regressor per action predicts
the final (clipped) score; policy = argmin(pred + lam * time). lam is chosen by family-grouped CV on TRAIN only.
"""
import json
import os

import numpy as np

from .labels import load
from .probe import N_PROBE

ACTIONS = ["QW", "QQS", "both"]
DEFAULT_GBM = {"max_depth": 3, "max_iter": 250, "learning_rate": 0.05, "min_samples_leaf": 10,
               "l2_regularization": 1.0, "random_state": 0}


def load_descriptors(path):
    z = np.load(path)
    return dict(zip(z["ids"].tolist(), z["desc"]))


def features(desc_vec, probe, n_star, t_prep, n_sharp):
    pr = [np.nan] * N_PROBE if probe is None else [np.nan if v is None else v for v in probe]
    return np.r_[desc_vec, pr, np.log(n_star), t_prep, n_sharp or 0, float(np.isnan(pr[0]))]


def table(qw_path, lab_path, desc):
    """QuadWild labels (with stored probe) + QQS labels -> X, S (score per action), T (time per action + QQS alone),
    Q (QQS score), ids."""
    lab = load(lab_path)
    X, S, T, Q, ids = [], [], [], [], []
    if not os.path.exists(qw_path):
        return [np.zeros((0, 1))] * 4 + [[]]
    for l in open(qw_path, encoding="utf-8"):
        r = json.loads(l)
        sid = r["solid_id"]
        if r.get("msg") == "memory" or sid not in desc or "qqs_default" not in lab.get(sid, {}):
            continue
        q = lab[sid]["qqs_default"]
        n_star = q.get("n_quads") or 2500
        s_qw, s_q = r.get("score", 10.0), q["score"]
        t_prep = r.get("t_prep", 10.0) or 10.0
        t_fin = max((r.get("t_all") or 30.0) - t_prep, 0.5)
        t_q = q.get("time_s", np.nan) / 2 if q.get("valid") else 60.0
        X.append(features(desc[sid], r.get("probe"), n_star, t_prep, r.get("n_sharp", 0)))
        S.append([s_qw, s_q, min(s_qw, s_q)])
        T.append([t_prep + t_fin, t_prep + t_q, t_prep + t_fin + t_q, t_q])
        Q.append(s_q); ids.append(sid)
    return np.array(X), np.array(S), np.array(T), np.array(Q), ids


def fit(X, S, gbm=None, clip=2.0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    params = {**DEFAULT_GBM, **(gbm or {})}
    return [HistGradientBoostingRegressor(**params).fit(X, np.minimum(S[:, a], clip)) for a in range(S.shape[1])]


def policy(models, X, S, T, lam):
    P = np.stack([m.predict(X) for m in models], 1)
    a = np.argmin(P + lam * T[:, :3], 1)
    r = np.arange(len(S))
    return S[r, a], T[r, a], a


def cv_lambda(X, S, T, groups, lams, k=5, tol=0.02, gbm=None, clip=2.0):
    """Family-grouped k-fold CV on TRAIN: largest lam whose CV mean score is within `tol` of the best."""
    from sklearn.model_selection import GroupKFold
    res = {l: [[], []] for l in lams}
    for tr, te in GroupKFold(k).split(X, groups=groups):
        ms = fit(X[tr], S[tr], gbm, clip)
        for l in lams:
            v, t, _ = policy(ms, X[te], S[te], T[te], l)
            res[l][0] += list(v); res[l][1] += list(t)
    out = [(l, np.mean(v), np.mean(t)) for l, (v, t) in res.items()]
    best = min(o[1] for o in out)
    return max(o[0] for o in out if o[1] <= best * (1 + tol)), out


def decide(models, x, lam, expected_times):
    p = np.array([m.predict(x[None])[0] for m in models])
    return int(np.argmin(p + lam * np.asarray(expected_times))), p


def save(path, models, lam, meta=None):
    import joblib
    joblib.dump({"models": models, "lam": lam, "actions": ACTIONS, "meta": meta or {}}, path)


def load_policy(path):
    import joblib
    return joblib.load(path)
