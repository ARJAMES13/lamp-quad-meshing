"""Mesh every part of a split with Gmsh QQS (and optionally Blossom); append one JSONL row per (part, method).

The QQS rows define each part's element budget n* (used by scripts/make_qw_labels.py and the policy).
Resumable: rows already in --out are skipped (crash/timeout/out-of-memory rows are retried). Each task runs in its
own process, started only while enough RAM is free.
Example: python scripts/make_labels.py --part test --methods qqs_default --out results/labels/my_test.jsonl
"""
import argparse
import json
import os

from _common import rel

RERUN_STATUSES = {"crash", "timeout", "infra_error"}
METHODS = "qqs_default, gmsh_blossom"


def task(tid, solid_id, path, method):
    from lamp_mesh.labelling import evaluate
    return evaluate(solid_id, path, method)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-config", default=rel("configs/data.toml"), help="dataset roots (TOML)")
    ap.add_argument("--split", default=rel("configs/splits/v1.json"))
    ap.add_argument("--part", default="test", choices=["train", "val", "test", "test_ood"])
    ap.add_argument("--methods", default="qqs_default", help="comma-separated: " + METHODS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N parts (0 = all)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--procs", type=int, default=2, help="max parallel worker processes")
    ap.add_argument("--timeout", type=float, default=900, help="seconds per (part, method)")
    a = ap.parse_args()

    from lamp_mesh.data import load_config, resolve
    from lamp_mesh.gated import run_tasks
    cfg = load_config(a.data_config)
    split = json.load(open(a.split))
    ids = split[a.part][a.offset:]
    if a.limit:
        ids = ids[:a.limit]
    methods = a.methods.split(",")
    kept, done = [], set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            if r.get("status") in RERUN_STATUSES:
                continue
            kept.append(line.rstrip("\n") + "\n")
            done.add((r["solid_id"], r["method"]))
        with open(a.out, "w") as fh:          # drop rows that will be retried
            fh.writelines(kept)
    todo = [(sid, m) for sid in ids for m in methods if (sid, m) not in done]
    print(f"{len(todo)} (part, method) tasks to run ({len(done)} done)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fh = open(a.out, "a")

    def on_result(args, res):
        _, sid, _, m = args[:4]
        if res[0] == "ok":
            row = res[1]
        else:
            st = "timeout" if "timeout" in str(res[1]) else "crash"
            row = {"solid_id": sid, "method": m, "status": st, "valid": False, "score": 10.0}
        row.pop("msg", None)
        fh.write(json.dumps(row) + "\n")
        fh.flush()

    tasks = [(f"{sid}|{m}", sid, resolve(sid, cfg), m) for sid, m in todo]
    run_tasks(task, tasks, on_result, max_procs=a.procs, timeout_s=a.timeout)
    fh.close()


if __name__ == "__main__":
    main()
