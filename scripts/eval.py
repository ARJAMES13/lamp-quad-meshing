"""Evaluate LAMP on the frozen TEST set and the HexMe OOD set against QQS alone.

Prints paired statistics for QQS alone, QuadWild alone, run-both-keep-best and LAMP (lam = 0 and the CV-chosen lam),
the action mix, the AUC of detecting "QuadWild worse than QQS", and a per-metric breakdown.
Without --model the policy is trained first (deterministic, well under a minute).
Examples:
  python scripts/eval.py
  python scripts/eval.py --model checkpoints/lamp.joblib
"""
import argparse

from _common import load_cfg, rel, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=rel("configs/lamp.toml"), help="policy config (TOML)")
    ap.add_argument("--model", default=None, help="saved policy from scripts/train.py (default: train now)")
    ap.add_argument("--sets", default="test,ood", help="comma-separated subset of: test, ood")
    a = ap.parse_args()

    import json

    import numpy as np
    from sklearn.metrics import roc_auc_score
    from lamp_mesh import policy as P
    from lamp_mesh.labels import load
    from train import train
    cfg = load_cfg(a.config)
    paths = cfg["paths"]
    if a.model:
        pol = P.load_policy(a.model); models, lam = pol["models"], pol["lam"]
    else:
        models, lam, _, _ = train(cfg)
    desc = P.load_descriptors(rel(paths["descriptors"]))
    keys = ["irregular_frac", "bad_frac", "sj_p05", "hausdorff_rel", "feat_dist_rel", "chamfer_rel"]
    for nm in a.sets.split(","):
        qwp, labp = rel(paths[f"{nm}_qw"]), rel(paths[f"{nm}_qqs"])
        X, S, T, Q, ids = P.table(qwp, labp, desc)
        if len(X) == 0:
            continue
        print(f"{nm.upper()} (n={len(X)})")
        r = lambda name, v, t: report(name, v, Q, t, delta="d", tdig=1)
        r("QQS alone", Q, T[:, 3])
        r("QW alone", S[:, 0], T[:, 0])
        r("run both, keep best", S[:, 2], T[:, 2])
        for l in sorted({0.0, lam}):
            v, t, act = P.policy(models, X, S, T, l)
            r(f"LAMP v2 lam={l}" + (" (CV-chosen)" if l == lam else ""), v, t)
            if l == lam:
                print("    actions:", {P.ACTIONS[k]: round(float(np.mean(act == k)), 2) for k in range(3)})
                act_star = act
        y = (S[:, 0] > S[:, 1] + 0.05).astype(int)
        if 0 < y.sum() < len(y):
            pred = models[0].predict(X) - models[1].predict(X)
            print(f"    'QW worse than QQS' detection AUC: {roc_auc_score(y, pred):.3f}  (positives {y.sum()}/{len(y)})")
        # per-metric breakdown of the kept meshes (parts where both QQS and LAMP's mesh are valid)
        lab = load(labp)
        qw = {json.loads(l)["solid_id"]: json.loads(l) for l in open(qwp, encoding="utf-8")}
        qqs = [lab[s]["qqs_default"] for s in ids]
        kept = []
        for i, s in enumerate(ids):
            cand = [qw[s]] if act_star[i] == 0 else [qqs[i]] if act_star[i] == 1 else [qw[s], qqs[i]]
            kept.append(min(cand, key=lambda m: m.get("score", 10.0)))
        ok = [i for i in range(len(ids)) if kept[i].get("valid") and qqs[i].get("valid")]
        print(f"    component means at lam*={lam} over {len(ok)} parts valid for both (lower is better except sj_p05):")
        for name, rows in (("QQS alone", qqs), ("LAMP v2", kept)):
            print(f"      {name:10s}" + "".join(f"  {k} {np.mean([rows[i][k] for i in ok]):.5f}" for k in keys))


if __name__ == "__main__":
    main()
