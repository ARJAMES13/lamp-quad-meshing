"""Train LAMP on TRAIN labels, choose lambda by family-grouped CV on TRAIN, save the policy.

Example: python scripts/train.py --out checkpoints/lamp.joblib
"""
import argparse
import json
import os

from _common import load_cfg, rel


def train(cfg, verbose=True):
    from lamp_mesh import policy as P
    pc, paths = cfg["policy"], cfg["paths"]
    desc = P.load_descriptors(rel(paths["descriptors"]))
    fam = json.load(open(rel(paths["split"])))["_family"]
    Xtr, Str, Ttr, _, idtr = P.table(rel(paths["train_qw"]), rel(paths["train_qqs"]), desc)
    if verbose:
        print(f"train labels: {len(Xtr)}")
    if len(Xtr) < 40:
        raise SystemExit("not enough train labels yet")
    lam, cvout = P.cv_lambda(Xtr, Str, Ttr, [fam.get(s, s) for s in idtr], pc["lambdas"], pc["cv_folds"],
                             pc["cv_tolerance"], pc.get("gbm"), pc["clip"])
    if verbose:
        print("CV (lam, mean score, mean time):", [(l, round(a, 4), round(b, 1)) for l, a, b in cvout], "-> lam*", lam)
    models = P.fit(Xtr, Str, pc.get("gbm"), pc["clip"])
    return models, lam, cvout, len(Xtr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=rel("configs/lamp.toml"), help="policy config (TOML)")
    ap.add_argument("--out", default=rel("checkpoints/lamp.joblib"), help="where to save the policy")
    a = ap.parse_args()
    from lamp_mesh import policy as P
    models, lam, cvout, n = train(load_cfg(a.config))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    P.save(a.out, models, lam, {"n_train": n, "cv": [(l, float(s), float(t)) for l, s, t in cvout]})
    print("saved", a.out, "(lam* =", lam, ")")


if __name__ == "__main__":
    main()
