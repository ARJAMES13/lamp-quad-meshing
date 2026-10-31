"""Regenerate the README figures from the committed labels (and, for the teaser, one inference run).

  results/figures/forest_test.png   Δ mean composite score vs QQS with 95 % CI, frozen test and OOD sets
  results/figures/teaser.png        input CAD triangulation -> output quad mesh (needs --step and --model)
The learning-curve figure comes from scripts/learning_curve.py.
Example: python scripts/make_figures.py --step part.step --model checkpoints/lamp.joblib
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

from _common import ROOT, load_cfg, rel


def forest(out, cfg):
    import matplotlib.pyplot as plt
    import numpy as np
    from lamp_mesh import policy as P
    from lamp_mesh.scoring import paired_bootstrap
    from train import train
    paths = cfg["paths"]
    models, lam, _, _ = train(cfg, verbose=False)
    desc = P.load_descriptors(rel(paths["descriptors"]))
    fig, axs = plt.subplots(1, 2, figsize=(10, 2.9), sharey=True)
    for ax, nm in zip(axs, ("test", "ood")):
        X, S, T, Q, _ = P.table(rel(paths[f"{nm}_qw"]), rel(paths[f"{nm}_qqs"]), desc)
        v, t, _ = P.policy(models, X, S, T, lam)
        rows = [("QuadWild alone", S[:, 0], T[:, 0]), (f"LAMP (λ={lam})", v, t), ("run both, keep best", S[:, 2], T[:, 2])]
        for k, (name, s, tt) in enumerate(rows[::-1]):
            md, lo, hi = paired_bootstrap(s, Q, n=10000)
            w, l_ = int((s < Q - .005).sum()), int((s > Q + .005).sum())
            c = "#1b9e77" if "LAMP" in name else "#555555"
            ax.plot([lo, hi], [k, k], color=c, lw=2.5); ax.plot(md, k, "o", color=c, ms=7)
            ax.text(0.37, k, f"{md:+.3f}  W/L {w}/{l_}  {np.nanmean(tt):.0f} s", va="center", fontsize=8)
        ax.axvline(0, color="#2166ac", lw=1.2)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=9)
        ax.set_xlim(-0.15, 0.75)
        ax.set_title(f"{'frozen test' if nm == 'test' else 'HexMe OOD'} ({len(Q)} parts; QQS {np.nanmean(T[:, 3]):.1f} s)",
                     fontsize=9)
        ax.set_xlabel("Δ mean score vs QQS (95 % CI); left is better", fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def _draw(ax, V, faces, title, color):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    polys = [V[f] for f in faces]
    ax.add_collection3d(Poly3DCollection(polys, facecolor=color, edgecolor="#222222", linewidths=0.15, alpha=1.0))
    lo, hi = V.min(0), V.max(0); c = (lo + hi) / 2; r = (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.view_init(elev=28, azim=-55); ax.set_box_aspect(None, zoom=1.3); ax.set_axis_off(); ax.set_title(title, fontsize=10)


def teaser(out, step, model):
    import matplotlib.pyplot as plt
    import numpy as np
    from lamp_mesh.geometry import build_surface
    from lamp_mesh.meshers import _read_poly_obj
    with tempfile.TemporaryDirectory() as td:
        obj, js = os.path.join(td, "out.obj"), os.path.join(td, "summary.json")
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "infer.py"), "--step", step, "--model", model,
                        "--out", obj, "--summary", js], capture_output=True, text=True, check=True)
        V, faces = _read_poly_obj(obj)
        info = json.load(open(js, encoding="utf-8"))
    s = build_surface(step, rel_h=0.012)
    fig = plt.figure(figsize=(9, 3.6))
    _draw(fig.add_subplot(1, 2, 1, projection="3d"), s.V, list(s.F), "input CAD (reference triangulation)", "#d9d9d9")
    _draw(fig.add_subplot(1, 2, 2, projection="3d"), np.asarray(V), faces,
          f"LAMP output: {info['n_quads']} quads, kept {info['kept']} (action {info['action']})", "#a6cee3")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.92, wspace=0)
    fig.savefig(out, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=rel("configs/lamp.toml"))
    ap.add_argument("--out-dir", default=rel("results/figures"))
    ap.add_argument("--step", default=None, help="CAD part for the teaser (skipped if omitted)")
    ap.add_argument("--model", default=None, help="saved policy for the teaser")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    os.makedirs(a.out_dir, exist_ok=True)
    cfg = load_cfg(a.config)
    forest(os.path.join(a.out_dir, "forest.png"), cfg)
    if a.step and a.model:
        teaser(os.path.join(a.out_dir, "teaser.png"), a.step, a.model)
    print("figures written to", a.out_dir)


if __name__ == "__main__":
    main()
