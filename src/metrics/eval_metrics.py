import sys, os, importlib
from pathlib import Path
root = Path(__file__).resolve().parent.parent.parent
eval_dir = str(root / "02_distillation" / "evaluation_suite")
if eval_dir not in sys.path: sys.path.insert(0, eval_dir)
_m = importlib.import_module("06_eval_metrics")
globals().update({k: v for k, v in _m.__dict__.items() if not k.startswith("_")})
