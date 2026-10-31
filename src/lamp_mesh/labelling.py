"""Mesh one part with Gmsh and record its a-priori metrics (one JSONL row per (part, method)).

Methods
  qqs_default       Gmsh QQS with its own size map, two passes (budget calibration); defines the budget n*
  gmsh_blossom      Gmsh frontal-Delaunay for quads + Blossom recombination
"""
import time
import traceback

CFG = {"rel_h": 0.012, "rel_edge": 0.015}
QQS_BG_METHOD = 4   # QuadqsSizemapMethod for external size fields


def evaluate(solid_id, path, method):
    """Return one result row; failures are recorded (status != ok), never raised."""
    from .geometry import ImportFail, build_surface
    from .meshers import run_gmsh_quads
    from .metrics import quad_metrics
    from .qqs import two_pass
    from .scoring import composite_score
    t0 = time.perf_counter()
    try:
        s = build_surface(path, rel_h=CFG["rel_h"])
    except ImportFail as e:
        return {"solid_id": solid_id, "method": method, "status": "import_fail", "valid": False, "msg": str(e)[:200]}
    t_surf = time.perf_counter() - t0
    m = method
    row = {"solid_id": solid_id, "method": m, "n_ref_verts": int(len(s.V))}
    tf = time.perf_counter()
    try:
        if m == "qqs_default":
            tp = two_pass(path, s, None, method=QQS_BG_METHOD)
            row.update({"tp": tp, "valid": bool(tp["p2"].get("valid") or tp["p1"].get("valid")),
                        "status": "ok" if tp["p1"].get("valid") else "extract_fail"})
        elif m == "gmsh_blossom":
            row.update(quad_metrics(run_gmsh_quads(path, s, CFG["rel_edge"], 8), s))
        else:
            raise ValueError(f"unknown method {m}")
    except MemoryError:   # out of RAM: infrastructure failure, rerun later, never a method loss
        row.update({"status": "infra_error", "valid": False, "msg": traceback.format_exc()[-300:]})
    except Exception:  # noqa: BLE001 — recorded as a failure of this method on this solid
        row.update({"status": "extract_fail", "valid": False, "msg": traceback.format_exc()[-300:]})
    row["field_time_s"] = 0.0
    row["total_time_s"] = t_surf + (time.perf_counter() - tf)
    # QQS rows are scored later at the per-part budget n* (labels.load); everything else is scored now
    row["score"] = None if "tp" in row else composite_score(row)
    return row
