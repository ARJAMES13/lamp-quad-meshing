"""Run QuadWild + Bi-MDF on every part of a QQS label file at that part's budget n*, store metrics + probe.

Each row holds the a-priori metrics of the budget-calibrated QuadWild mesh, the probe-stage time (t_prep), the
total time of all quantisation runs (t_all), the calibration history and the 10-D patch-layout probe used by LAMP.
Resumable and memory-gated; then prints paired statistics vs QQS.
Example: python scripts/make_qw_labels.py --labels results/labels/test_v1.jsonl --out results/labels/my_qw_test.jsonl
"""
import argparse
import json
import os
import tempfile

from _common import load_cfg, rel


def qw_task(sid, path, n_star, qcfg, keep_dir):
    import numpy as np
    from lamp_mesh.geometry import build_surface
    from lamp_mesh.labelling import CFG
    from lamp_mesh.meshers import QuadResult
    from lamp_mesh.metrics import quad_metrics
    from lamp_mesh.probe import probe_workdir
    from lamp_mesh.quadwild import QuadWild, calibrated
    from lamp_mesh.scoring import composite_score
    s = build_surface(path, rel_h=CFG["rel_h"])
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        work = os.path.join(keep_dir, sid.replace("/", "_")) if keep_dir else td
        qw = QuadWild(s.V, s.F, s.feat_edges, work, sharp_angle=qcfg["sharp_angle"])
        pr = probe_workdir(work, n_star)
        V, Q, T, n, t, t_q, hist = calibrated(qw, n_star, qcfg["calibration_tol"], qcfg["max_runs"])
    t_all = qw.t_prep + t_q
    probe = None if pr is None else [float(v) for v in pr]
    if V is None:
        return {"solid_id": sid, "valid": False, "score": 10.0, "hist": hist, "t_all": t_all, "probe": probe}
    m = quad_metrics(QuadResult("ok", V, Q, T, t), s)
    m["score"] = composite_score(m) if m.get("valid") else 10.0
    # one-run time = prep + one quantisation (calibration reruns counted in t_all)
    m.update({"solid_id": sid, "n": int(n), "n_star": float(n_star), "hist": hist, "t_prep": qw.t_prep,
              "t_one": qw.t_prep + t, "t_all": t_all, "n_sharp": qw.n_sharp, "probe": probe})
    return {k: v for k, v in m.items() if not isinstance(v, np.ndarray)}


def stats(labels, out):
    import numpy as np
    from scipy.stats import binomtest
    from lamp_mesh.labels import load
    from lamp_mesh.scoring import paired_bootstrap
    lab = load(labels)
    rows = {r["solid_id"]: r for r in (json.loads(l) for l in open(out, encoding="utf-8"))}
    sids = [s for s in rows if s in lab and "qqs_default" in lab[s]]
    q = np.array([lab[s]["qqs_default"]["score"] for s in sids])
    w = np.array([rows[s].get("score", 10.0) for s in sids])
    md, lo, hi = paired_bootstrap(w, q, n=10000)
    W, L = int((w < q - .005).sum()), int((w > q + .005).sum())
    ratio = np.array([rows[s].get("n", 0) / rows[s].get("n_star", 1) for s in sids if rows[s].get("valid")])
    print(f"n={len(sids)}  QQS mean {q.mean():.4f} median {np.median(q):.4f} | QuadWild mean {w.mean():.4f} median {np.median(w):.4f}")
    print(f"Δmean {md:+.4f} [{lo:+.4f},{hi:+.4f}]  W/L {W}/{L}  sign p {binomtest(W, W + L, .5).pvalue:.1e}")
    print(f"elements n/n*: median {np.median(ratio):.3f} [p10 {np.quantile(ratio, .1):.3f}, p90 {np.quantile(ratio, .9):.3f}]  "
          f"failures {(w >= 10).sum()}  time one-run median {np.nanmedian([rows[s].get('t_one', np.nan) for s in sids]):.1f}s "
          f"(QQS {np.nanmedian([lab[s]['qqs_default'].get('time_s', np.nan) / 2 for s in sids]):.1f}s)")
    for k in ("irregular_frac", "bad_frac", "sj_p05", "hausdorff_rel", "feat_dist_rel"):
        a = np.nanmean([lab[s]["qqs_default"].get(k, np.nan) for s in sids if rows[s].get("valid")])
        b = np.nanmean([rows[s].get(k, np.nan) for s in sids if rows[s].get("valid")])
        print(f"  {k:15s} QQS {a:.4f}  QuadWild {b:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="QQS label file (defines the parts and their budget n*)")
    ap.add_argument("--out", required=True, help="QuadWild label file to append to")
    ap.add_argument("--config", default=rel("configs/lamp.toml"))
    ap.add_argument("--data-config", default=rel("configs/data.toml"), help="dataset roots (TOML)")
    ap.add_argument("--procs", type=int, default=2, help="max parallel worker processes")
    ap.add_argument("--timeout", type=float, default=900, help="seconds per part")
    ap.add_argument("--keep-dir", default=None, help="keep QuadWild working files here (default: temporary)")
    ap.add_argument("--stats-only", action="store_true", help="only print paired statistics of --out vs QQS")
    a = ap.parse_args()
    if not a.stats_only:
        from lamp_mesh.data import load_config, resolve
        from lamp_mesh.gated import run_tasks
        from lamp_mesh.labels import load
        qcfg = load_config(a.config)["quadwild"]
        dcfg = load_config(a.data_config)
        lab = load(a.labels)
        done = {json.loads(l)["solid_id"] for l in open(a.out, encoding="utf-8")} if os.path.exists(a.out) else set()
        tasks = [(s, resolve(s, dcfg), lab[s]["qqs_default"]["n_quads"], qcfg, a.keep_dir) for s in lab
                 if s not in done and lab[s].get("qqs_default", {}).get("valid") and lab[s]["qqs_default"].get("n_quads")]
        print(len(tasks), "tasks", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

        def on_result(args, res):
            status, payload = res
            row = payload if status == "ok" and isinstance(payload, dict) else \
                {"solid_id": args[0], "valid": False, "score": 10.0, "status": status,
                 "msg": "memory" if payload == "memory" else "failed"}
            row = {k: v for k, v in row.items() if isinstance(v, (int, float, str, list, bool)) or v is None}
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        run_tasks(qw_task, tasks, on_result, max_procs=a.procs, timeout_s=a.timeout)
    stats(a.labels, a.out)


if __name__ == "__main__":
    main()
