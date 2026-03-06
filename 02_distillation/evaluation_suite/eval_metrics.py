"""
eval_metrics.py — Standard Python import alias for 06_eval_metrics.py
"""
import sys
import os
import importlib

_cur_dir = os.path.dirname(os.path.abspath(__file__))
if _cur_dir not in sys.path:
    sys.path.insert(0, _cur_dir)

_mod = importlib.import_module("06_eval_metrics")
globals().update({k: v for k, v in _mod.__dict__.items() if not k.startswith("_")})
