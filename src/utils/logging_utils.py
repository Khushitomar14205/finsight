"""Structured logging. Never log secrets — callers must not pass them."""
from __future__ import annotations
import logging, sys
from pathlib import Path

def get_logger(name: str = "finsight", log_dir: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_dir / "finsight.log"); fh.setFormatter(fmt); logger.addHandler(fh)
        except Exception:
            pass
    logger.propagate = False
    return logger

def redact(text: str, secrets: list[str]) -> str:
    out = text
    for s in secrets:
        if s and len(s) >= 6:
            out = out.replace(s, "***REDACTED***")
    return out
