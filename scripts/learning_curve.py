"""LAMP learning curve: train on the first N QuadWild train labels (in labelling order), evaluate on TEST and OOD.

For each N: lam by family-grouped CV on those N parts, then Δ mean vs QQS [95 % CI] and the AUC of detecting
"QuadWild worse than QQS by > 0.05". Optionally saves a figure.
Example: python scripts/learning_curve.py --sizes 71,204,387 --fig results/figures/learning_curve.png
"""
import argparse
import json
import os
import tempfile

from _common import load_cfg, rel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=rel("configs/lamp.toml"))
    ap.add_argument("--sizes", default="71,204,387", help="comma-separated numbers of train labels")
    ap.add_argument("--fig", default=None, help="save a figure here")
    a = ap.parse_args()

    import numpy as np
    from sklearn.metrics import roc_auc_score
    from lamp_mesh import policy as P
    from lamp_mesh.scoring import paired_bootstrap
    cfg = load_cfg(a.config)
    pc, paths = cfg["policy"], cfg["paths"]
    desc = P.load_descriptors(rel(paths["descriptors"]))
    fam = json.load(open(rel(paths["split"])))["_family"]
    lines = open(rel(paths["train_qw"]), encoding="utf-8").readlines()
    sets = {nm: P.table(rel(paths[f"{nm}_qw"]), rel(paths[f"{nm}_qqs"]), desc)[:4] for nm in ("test", "ood")}
    res = {nm: [] for nm in sets}
    sizes = [int(x) for x in a.sizes.split(",")]
    for n in sizes:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl", encoding="utf-8") as f:
            f.writelines(lines[:n])
        X, S, T, _, ids = P.table(f.name, rel(paths["train_qqs"]), desc)
        os.unlink(f.name)
        lam, _ = P.cv_lambda(X, S, T, [fam.get(s, s) for s in ids], pc["lambdas"], pc["cv_folds"], pc["cv_tolerance"],
                             pc.get("gbm"), pc["clip"])
        models = P.fit(X, S, pc.get("gbm"), pc["clip"])
        line = f"N={n:4d} (used {len(X)}) lam*={lam}"
        for nm, (Xe, Se, Te, Qe) in sets.items():
            v, _, _ = P.policy(models, Xe, Se, Te, lam)
            md, lo, hi = paired_bootstrap(v, Qe, n=10000)
            y = (Se[:, 0] > Se[:, 1] + 0.05).astype(int)
            auc = roc_auc_score(y, models[0].predict(Xe) - models[1].predict(Xe))
            res[nm].append((md, lo, hi, auc))
            line += f" | {nm.upper()} AUC {auc:.3f} d {md:+.4f} [{lo:+.4f},{hi:+.4f}]"
        print(line, flush=True)
    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
        for nm, c in (("test", "#1b9e77"), ("ood", "#7570b3")):
            r = np.array(res[nm])
            axs[0].plot(sizes, r[:, 3], "o-", color=c, label=nm.upper())
            axs[1].errorbar(sizes, r[:, 0], yerr=[r[:, 0] - r[:, 1], r[:, 2] - r[:, 0]], fmt="o-", color=c, capsize=3,
                            label=nm.upper())
        axs[0].axhline(0.5, color="#999999", lw=1, ls=":"); axs[0].set_ylabel("AUC: 'QuadWild worse than QQS'")
        axs[1].axhline(0, color="#2166ac", lw=1); axs[1].set_ylabel("Δ mean score vs QQS (95 % CI)")
        for ax in axs:
            ax.set_xlabel("train labels"); ax.legend(fontsize=8, frameon=False)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        fig.tight_layout(); fig.savefig(a.fig, dpi=150); plt.close(fig)
        print("saved", a.fig)


if __name__ == "__main__":
    main()
