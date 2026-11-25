import argparse
import glob
import os
import shutil
import time

import jconfig as C
import explanation_stability as ES
from jutils import RunLock, get_logger, read_json, release_gpu, run_dir_for, write_json, write_npz

LOG = None
ALL_SEEDS = list(C.SEEDS_10)
EXPL_DIR = C.DIRS["explain"]


def out_dir(d, c, m):
    return os.path.join(EXPL_DIR, d, c, m.replace("+", "_plus_"))


def jobs(datasets, conditions, models):
    return [(d, c, m, s) for d in datasets for c in conditions for m in models for s in NEW_SEEDS]


# ---------------------------------------------------------------------
# Stage 1 and stage 2 
# ---------------------------------------------------------------------

def stage1(d, c, m, s, bundle, device):
    od = out_dir(d, c, m)
    out = os.path.join(od, f"seed{s}.npz")
    if os.path.exists(out):
        return "done"
    with RunLock(os.path.join(od, f"lock_seed{s}")) as lk:
        if not lk.acquired or os.path.exists(out):
            return "locked"
        t0 = time.perf_counter()
        res = ES.explain_run(d, c, m, s, bundle, device)
        if res is None:
            return "missing model"
        arr, meta = res
        meta.update({"wall_s": time.perf_counter() - t0, "lime_version": ES.LIME_VERSION,
                     "lime_sampling": "around the explained node (sample_around_instance=True)",
                     "added_by": "explain_ten_seeds.py (C7)", "repeat_floor": "not run for this seed (floor from seeds 42, 52, 62)"})
        write_json(os.path.join(od, f"seed{s}.json"), meta)   # json first: npz marks the run as finished
        write_npz(out, **arr)
        LOG.info(f"stage 1 {d} {c} {m} seed{s} done in {meta['wall_s'] / 60:.1f} min")
        return "ok"


def stage2(d, c, m, s, bundle, device):
    if m not in ES.GRAD_MODELS:
        return "n/a"
    od = out_dir(d, c, m)
    meta = read_json(os.path.join(od, f"seed{s}.json")) or {}
    if meta.get("gnnx_version") == ES.GNNX_VERSION:
        return "done"
    if not os.path.exists(os.path.join(od, f"seed{s}.npz")):
        return "no stage 1"
    with RunLock(os.path.join(od, f"gnnxlock_seed{s}")) as lk:
        if not lk.acquired:
            return "locked"
        t0 = time.perf_counter()
        st = ES.gnnx_run(d, c, m, s, bundle, device)
        if st == "ok":
            LOG.info(f"stage 2 {d} {c} {m} seed{s} done in {(time.perf_counter() - t0) / 60:.1f} min")
        return st


# ---------------------------------------------------------------------
# Tables over ten seeds
# ---------------------------------------------------------------------

def summarise_ten():
    keep3 = os.path.join(EXPL_DIR, "summary_3seeds")
    out10 = os.path.join(EXPL_DIR, "summary_10seeds")
    current = glob.glob(os.path.join(EXPL_DIR, "summary_*.csv"))
    if not os.path.isdir(keep3):
        os.makedirs(keep3)
        for f in current:
            shutil.copy2(f, keep3)
        LOG.info(f"Kept a copy of the 3-seed tables in {keep3}")
    C.SEEDS_EXPLAIN = ALL_SEEDS         
    try:
        ES.summarise()
    finally:
        C.SEEDS_EXPLAIN = OLD_SEEDS
    os.makedirs(out10, exist_ok=True)
    for f in glob.glob(os.path.join(EXPL_DIR, "summary_*.csv")):
        shutil.move(f, os.path.join(out10, os.path.basename(f)))
    for f in glob.glob(os.path.join(keep3, "summary_*.csv")):   
        shutil.copy2(f, EXPL_DIR)
    LOG.info(f"10-seed tables written to {out10}")


# ---------------------------------------------------------------------

def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", choices=C.DATASETS, default=C.DATASETS)
    ap.add_argument("--model", nargs="*", choices=C.EXPLAIN_MODELS, default=C.EXPLAIN_MODELS)
    ap.add_argument("--condition", nargs="*", choices=ES.EXPLAIN_CONDITIONS, default=ES.EXPLAIN_CONDITIONS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check", action="store_true", help="preflight only")
    ap.add_argument("--summarise", action="store_true", help="10-seed tables only")
    args, _ = ap.parse_known_args()
    C.ensure_dirs()
    LOG = get_logger("explain_ten_seeds", C.DIRS["logs"])
    ES.LOG = LOG
    if args.summarise:
        summarise_ten()
        return

    ES.E["convergence_nodes"] = 0        
    import torch
    from jdata import DatasetBundle
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    LOG.info(f"device {device}")
    t_all, n_ok, wall = time.perf_counter(), 0, 0.0
    todo = jobs(args.dataset, args.condition, args.model)
    for d in args.dataset:
        bundle = DatasetBundle(d)
        for c in args.condition:
            for m in args.model:
                for s in NEW_SEEDS:
                    for stage in (stage1, stage2):
                        t0 = time.perf_counter()
                        try:
                            st = stage(d, c, m, s, bundle, device)
                        except Exception as e:
                            LOG.exception(f"{stage.__name__} {d} {c} {m} seed{s} failed: {e}")
                            st = "failed"
                        release_gpu(device)
                        if st == "ok":
                            n_ok += 1
                            wall += time.perf_counter() - t0
                    done = sum(os.path.exists(os.path.join(out_dir(*j[:3]), f"seed{j[3]}.npz")) for j in todo)
                    LOG.info(f"progress: {done}/{len(todo)} stage-1 runs finished, "
                             f"{(time.perf_counter() - t_all) / 3600:.2f} h elapsed in this worker")
        del bundle
    left = [j for j in todo if not os.path.exists(os.path.join(out_dir(*j[:3]), f"seed{j[3]}.npz"))
            or (j[2] in ES.GRAD_MODELS and (read_json(os.path.join(out_dir(*j[:3]), f"seed{j[3]}.json")) or {})
                .get("gnnx_version") != ES.GNNX_VERSION)]
    LOG.info(f"This worker finished {n_ok} stages in {wall / 3600:.2f} h of explainer time")
    if not left:
        with RunLock(os.path.join(EXPL_DIR, "summary10_lock")) as lk:   # only one worker builds the tables
            if lk.acquired:
                summarise_ten()
    else:
        LOG.info(f"{len(left)} runs are still open (other workers or failures). Run again, or "
                 f"run -summarise once every worker has finished.")


if __name__ == "__main__":
    main()
