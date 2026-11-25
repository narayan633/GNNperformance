import datetime as _dt
import hashlib
import json
import logging
import os
import platform
import socket
import sys
import tempfile
import time

import numpy as np


# ---------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def write_json(path, obj):

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_npz(path, **arrays):

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".npz")
    os.close(fd)
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_hash(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=_json_default).encode()).hexdigest()


def now():
    return _dt.datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------
# Run locks 
# ---------------------------------------------------------------------

class RunLock:


    def __init__(self, run_dir, stale_hours=6.0):
        self.run_dir = run_dir
        self.path = os.path.join(run_dir, ".lock")
        self.stale_seconds = stale_hours * 3600
        self.acquired = False

    def __enter__(self):
        os.makedirs(self.run_dir, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as f:
                    f.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "time": now()}))
                self.acquired = True
                break
            except FileExistsError:
                try:
                    age = time.time() - os.path.getmtime(self.path)
                except FileNotFoundError:
                    continue
                if age > self.stale_seconds:
                    try:
                        os.remove(self.path)
                    except FileNotFoundError:
                        pass
                    continue
                break
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.acquired:
            try:
                os.remove(self.path)
            except FileNotFoundError:
                pass
        return False


def is_done(run_dir):
    return os.path.exists(os.path.join(run_dir, "DONE"))


def mark_done(run_dir, info=None):
    write_json(os.path.join(run_dir, "DONE"), {"time": now(), **(info or {})})


# ---------------------------------------------------------------------
# Seeding and environment
# ---------------------------------------------------------------------

def set_seed(seed, deterministic=True):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # PyG scatter operations have no deterministic CUDA kernel; warn only.
            torch.use_deterministic_algorithms(True, warn_only=True)
            import warnings
            warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")
    except ImportError:
        pass


def environment_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "host": socket.gethostname(), "time": now()}
    for mod in ("numpy", "scipy", "sklearn", "torch", "torch_geometric", "xgboost", "optuna", "shap", "lime"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            info[mod] = None
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except Exception:
        pass
    return info


def get_logger(name, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}_{os.getpid()}.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def run_dir_for(runs_root, exp, dataset, condition, model, seed):
    safe_model = model.replace("+", "_plus_")
    return os.path.join(runs_root, exp, dataset, condition, safe_model, f"seed{seed}")
