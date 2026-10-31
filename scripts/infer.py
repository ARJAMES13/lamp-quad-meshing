"""Quad-mesh one CAD part with LAMP and write the best mesh as OBJ.

1. QuadWild field + trace stage on the CAD triangulation with exact CAD sharp edges (the probe);
2. LAMP predicts, from the 15-D shape descriptor + patch-layout statistics, which finishing action is worth its
   time: QW (QuadWild quantisation at the budget) | QQS (Gmsh quasi-structured quads) | both;
3. the attempt with the lowest composite a-priori score is written.
Example: python scripts/infer.py --step part.step --model checkpoints/lamp.joblib --out part_quads.obj
"""
import argparse
import json
import tempfile
import time

from _common import load_cfg, rel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True, help="input CAD file (STEP/BREP, one solid)")
    ap.add_argument("--model", required=True, help="policy saved by scripts/train.py")
    ap.add_argument("--out", required=True, help="output quad mesh (.obj)")
    ap.add_argument("--summary", default=None, help="also write the JSON summary to this file")
    ap.add_argument("--config", default=rel("configs/lamp.toml"))
    ap.add_argument("--target-quads", type=int, default=None, help="element budget (default: config value)")
    ap.add_argument("--lam", type=float, default=None, help="override the saved time/quality trade-off")
    ap.add_argument("--action", default=None, choices=["QW", "QQS", "both"], help="force an action")
    a = ap.parse_args()

    from lamp_mesh import policy as P
    from lamp_mesh.catalog import shape_descriptor
    from lamp_mesh.geometry import build_surface
    from lamp_mesh.labelling import CFG
    from lamp_mesh.meshers import QuadResult, write_quad_obj
    from lamp_mesh.metrics import quad_metrics
    from lamp_mesh.probe import probe_workdir
    from lamp_mesh.qqs import run_qqs
    from lamp_mesh.quadwild import QuadWild, calibrated
    from lamp_mesh.scoring import composite_score

    cfg = load_cfg(a.config)
    icfg, qcfg = cfg["inference"], cfg["quadwild"]
    pol = P.load_policy(a.model)
    lam = pol["lam"] if a.lam is None else a.lam
    n_star = a.target_quads or icfg["target_quads"]
    t0 = time.perf_counter()
    s = build_surface(a.step, rel_h=CFG["rel_h"])
    desc = shape_descriptor(a.step)
    if desc is None:
        raise SystemExit("input does not import as exactly one solid")

    def attempt(name, r):
        m = quad_metrics(r, s)
        return {"name": name, "result": r, "metrics": m, "score": composite_score(m)}

    tried = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as work:
        qw = QuadWild(s.V, s.F, s.feat_edges, work, sharp_angle=qcfg["sharp_angle"])
        pr = probe_workdir(work, n_star)
        x = P.features(desc, pr, n_star, qw.t_prep, qw.n_sharp)
        act = P.ACTIONS.index(a.action) if a.action else P.decide(pol["models"], x, lam, icfg["expected_time_s"])[0]
        if P.ACTIONS[act] in ("QW", "both"):
            V, Q, T, n, t, _, _ = calibrated(qw, n_star, qcfg["calibration_tol"], qcfg["max_runs"])
            if V is not None:
                tried.append(attempt("QuadWild", QuadResult("ok", V, Q, T, t)))
    if P.ACTIONS[act] in ("QQS", "both") or not tried:
        tried.append(attempt("QQS", run_qqs(a.step, s, None, n_target=n_star)))
    best = min(tried, key=lambda t: t["score"])
    write_quad_obj(a.out, best["result"])
    summary = {"action": P.ACTIONS[act], "lam": lam, "target_quads": n_star,
               "probe_failed": pr is None, "attempts": {t["name"]: round(t["score"], 4) for t in tried},
               "kept": best["name"], "n_quads": int(len(best["result"].Q)),
               "wall_s": round(time.perf_counter() - t0, 2),
               "metrics": {k: round(v, 5) for k, v in best["metrics"].items() if isinstance(v, float)}}
    print(json.dumps(summary, indent=1))
    if a.summary:
        with open(a.summary, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
