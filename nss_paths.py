"""
Shared Mooney-split selection and output paths for the NSS pipeline.

Every NSS script that reads or writes the *cross-phase* outputs calls `select()`
at startup, which prompts once for the Mooney-window mode ("whole" or "halves")
so you can never run a script against the wrong version by forgetting a setting.

Folder layout (relative to the project root, where these scripts are run from):

    analysesresults/NSS/            shared, mode-independent
        FixMaps_full.pkl
        NSS_WithinPhase.pkl
        NSS_WithinPhase_LongFormat.csv
    analysesresults/NSS_<mode>/     per-mode  (mode = whole | halves)
        NSS_crossphase_descriptives.pkl
        NSS_CrossPhase_LongFormat.csv
        NSS_CrossPhase_LongFormat_centred.csv

Only the cross-phase step depends on the Mooney split, so FixMaps and the
within-phase results are shared across modes (and never rebuilt when you switch).

Non-interactive override: set the MOONEY_SPLIT environment variable to
"whole" or "halves" to skip the prompt (handy for scripted/batch runs).
"""

from pathlib import Path
import os
import sys

VALID = ("whole", "halves")
SHARED_DIR = Path("analysesresults/NSS")  # mode-independent: FixMaps + within-phase


def ask_mooney_split() -> str:
    """Return "whole"/"halves" from $MOONEY_SPLIT if set, otherwise prompt the user."""
    env = os.environ.get("MOONEY_SPLIT", "").strip().lower()
    if env in VALID:
        print(f"[nss_paths] MOONEY_SPLIT = {env}  (from environment)")
        return env
    if not (sys.stdin and sys.stdin.isatty()):
        raise SystemExit(
            "MOONEY_SPLIT is not set and there is no terminal to prompt on. "
            "Set the environment variable MOONEY_SPLIT=whole|halves and rerun."
        )
    while True:
        answer = input("\nWhich Mooney-window version? [w]hole / [h]alves: ").strip().lower()
        if answer in ("w", "whole"):
            return "whole"
        if answer in ("h", "halves"):
            return "halves"
        print("  Please type 'w' (whole) or 'h' (halves).")


def paths_for(mooney_split: str) -> dict:
    """Resolve every NSS output path for the given mode and ensure the folder exists."""
    if mooney_split not in VALID:
        raise ValueError(f"mooney_split must be one of {VALID}, got {mooney_split!r}")
    out_dir = Path(f"analysesresults/NSS_{mooney_split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "MOONEY_SPLIT": mooney_split,
        "SHARED_DIR": SHARED_DIR,
        "OUTPUT_DIR": out_dir,
        # shared (mode-independent)
        "FIXMAPS_PKL": SHARED_DIR / "FixMaps_full.pkl",
        "WITHIN_PKL": SHARED_DIR / "NSS_WithinPhase.pkl",
        "WITHIN_CSV": SHARED_DIR / "NSS_WithinPhase_LongFormat.csv",
        # per-mode (cross-phase)
        "CROSS_PKL": out_dir / "NSS_crossphase_descriptives.pkl",
        "CROSS_CSV": out_dir / "NSS_CrossPhase_LongFormat.csv",
        "CROSS_CENTRED_CSV": out_dir / "NSS_CrossPhase_LongFormat_centred.csv",
    }


def select() -> dict:
    """Prompt (or read $MOONEY_SPLIT) and return the resolved path dict."""
    return paths_for(ask_mooney_split())
