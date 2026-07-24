"""
Shared Mooney-split / trial-set selection and output paths for the NSS pipeline.

Every NSS script that reads or writes the *cross-phase* outputs calls `select()`
at startup, which prompts once for the Mooney-window mode ("whole" or "halves"),
once for the trial set ("all" or "experiment"-block-only trials) and once for the
blink mode ("filter" = original event-drop, or "interp" = PCHIP-interpolated blinks),
so you can never run a script against the wrong version by forgetting a setting.

Folder layout (relative to the project root, where these scripts are run from).
<suffix> concatenates the trial-set suffix ("" for all, "_exponly" for experiment,
"_extraonly" for extra) and then the blink-mode suffix ("" for filter, "_interp" for
interp). For example "", "_interp", "_exponly", "_exponly_interp":

    analysesresults/NSS<suffix>/            shared across Mooney modes
        FixMaps_full.pkl
        NSS_WithinPhase.pkl
        NSS_WithinPhase_LongFormat.csv
    analysesresults/NSS_<mode><suffix>/     per-mode  (mode = whole | halves)
        NSS_crossphase_descriptives.pkl
        NSS_CrossPhase_LongFormat.csv
        NSS_CrossPhase_LongFormat_centred.csv

Only the cross-phase step depends on the Mooney split, so FixMaps and the
within-phase results are shared across modes (and never rebuilt when you switch).
The trial set, however, changes the input fixations themselves (Extra-block
trials are dropped entirely), so everything is versioned by it: FixMaps, within-phase
and cross-phase.

Non-interactive override: set the MOONEY_SPLIT ("whole"/"halves"), TRIAL_SET
("all"/"experiment"/"extra") and BLINK_MODE ("filter"/"interp") environment
variables to skip the prompts (handy for scripted/batch runs).
"""

from pathlib import Path
import os
import sys

VALID = ("whole", "halves")
VALID_TRIAL_SET = ("all", "experiment", "extra")
VALID_BLINK = ("filter", "interp")

# Folder suffix per trial set (single source of truth, used both to resolve
# output paths and to name plot files, so the two never drift apart).
TRIAL_SET_SUFFIX = {"all": "", "experiment": "_exponly", "extra": "_extraonly"}

# Folder suffix per blink-handling mode of the Stage-1 parquet. "filter" (blinks
# dropped, original pipeline) gets no suffix so existing folders stay valid; "interp"
# (blinks PCHIP-interpolated) writes to *_interp folders so both can coexist on disk.
BLINK_SUFFIX = {"filter": "", "interp": "_interp"}


def _no_tty_exit(var: str, values: tuple) -> None:
    raise SystemExit(
        f"{var} is not set and there is no terminal to prompt on. "
        f"Set the environment variable {var}={'|'.join(values)} and rerun."
    )


def ask_mooney_split() -> str:
    """Return "whole"/"halves" from $MOONEY_SPLIT if set, otherwise prompt the user."""
    env = os.environ.get("MOONEY_SPLIT", "").strip().lower()
    if env in VALID:
        print(f"[NSSPaths] MOONEY_SPLIT = {env}  (from environment)")
        return env
    if not (sys.stdin and sys.stdin.isatty()):
        _no_tty_exit("MOONEY_SPLIT", VALID)
    while True:
        answer = input("\nWhich Mooney-window version? [w]hole / [h]alves: ").strip().lower()
        if answer in ("w", "whole"):
            return "whole"
        if answer in ("h", "halves"):
            return "halves"
        print("  Please type 'w' (whole) or 'h' (halves).")


def ask_trial_set() -> str:
    """Return "all"/"experiment"/"extra" from $TRIAL_SET if set, otherwise prompt the user."""
    env = os.environ.get("TRIAL_SET", "").strip().lower()
    if env in VALID_TRIAL_SET:
        print(f"[NSSPaths] TRIAL_SET = {env}  (from environment)")
        return env
    if not (sys.stdin and sys.stdin.isatty()):
        _no_tty_exit("TRIAL_SET", VALID_TRIAL_SET)
    while True:
        answer = input("Which trials? [a]ll / [e]xperiment-block only / e[x]tra-block only: ").strip().lower()
        if answer in ("a", "all"):
            return "all"
        if answer in ("e", "experiment"):
            return "experiment"
        if answer in ("x", "extra"):
            return "extra"
        print("  Please type 'a' (all), 'e' (experiment-block only) or 'x' (extra-block only).")


def ask_blink_mode() -> str:
    """Return "filter"/"interp" from $BLINK_MODE if set, otherwise prompt the user.

    Must match how Stage 1 built the parquet ("filter" = blinks dropped, the original
    pipeline; "interp" = blinks PCHIP-interpolated). It is not recorded in the parquet,
    so choosing it here both selects the *_interp output folder and lets NSS.py stamp it
    into the cache meta.
    """
    env = os.environ.get("BLINK_MODE", "").strip().lower()
    if env in VALID_BLINK:
        print(f"[NSSPaths] BLINK_MODE = {env}  (from environment)")
        return env
    if not (sys.stdin and sys.stdin.isatty()):
        _no_tty_exit("BLINK_MODE", VALID_BLINK)
    while True:
        answer = input("Which blink handling? [f]ilter (drop, original) / [i]nterp (PCHIP): ").strip().lower()
        if answer in ("f", "filter"):
            return "filter"
        if answer in ("i", "interp"):
            return "interp"
        print("  Please type 'f' (filter) or 'i' (interp).")


def paths_for(mooney_split: str, trial_set: str = "all", blink_mode: str = "filter") -> dict:
    """Resolve every NSS output path for the given mode/trial set/blink mode and ensure the folder exists."""
    if mooney_split not in VALID:
        raise ValueError(f"mooney_split must be one of {VALID}, got {mooney_split!r}")
    if trial_set not in VALID_TRIAL_SET:
        raise ValueError(f"trial_set must be one of {VALID_TRIAL_SET}, got {trial_set!r}")
    if blink_mode not in VALID_BLINK:
        raise ValueError(f"blink_mode must be one of {VALID_BLINK}, got {blink_mode!r}")
    suffix = TRIAL_SET_SUFFIX[trial_set] + BLINK_SUFFIX[blink_mode]  # trial set, then blink mode
    shared_dir = Path(f"analysesresults/NSS{suffix}")  # mode-independent: FixMaps + within-phase
    out_dir = Path(f"analysesresults/NSS_{mooney_split}{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "MOONEY_SPLIT": mooney_split,
        "TRIAL_SET": trial_set,
        "BLINK_MODE": blink_mode,
        "SHARED_DIR": shared_dir,
        "OUTPUT_DIR": out_dir,
        # shared across Mooney modes (but per trial set)
        "FIXMAPS_PKL": shared_dir / "FixMaps_full.pkl",
        "WITHIN_PKL": shared_dir / "NSS_WithinPhase.pkl",
        "WITHIN_CSV": shared_dir / "NSS_WithinPhase_LongFormat.csv",
        # per-mode (cross-phase)
        "CROSS_PKL": out_dir / "NSS_crossphase_descriptives.pkl",
        "CROSS_CSV": out_dir / "NSS_CrossPhase_LongFormat.csv",
        "CROSS_CENTRED_CSV": out_dir / "NSS_CrossPhase_LongFormat_centred.csv",
    }


def filter_trial_set(fixations_df, trial_set: str):
    """Restrict a fixations DataFrame (pandas, from the parquet) to the chosen trial set.

    "all" returns the frame untouched; "experiment" keeps only Experiment-block
    fixations and "extra" keeps only Extra-block fixations. Both single-block sets
    need the parquet's block_type column; rerun NSSExporter.py if it is missing.
    """
    if trial_set == "all":
        return fixations_df
    block = {"experiment": "Experiment", "extra": "Extra"}[trial_set]
    if "block_type" not in fixations_df.columns:
        raise SystemExit(
            f"TRIAL_SET={trial_set} needs the 'block_type' column in the fixations "
            "parquet. Rerun Scripts/Analysis/NSS/NSSExporter.py to regenerate it."
        )
    out = fixations_df[fixations_df["block_type"] == block]
    print(f"[NSSPaths] trial set '{trial_set}': kept {len(out)} / {len(fixations_df)} fixations ({block} block only)")
    return out


def select() -> dict:
    """Prompt (or read $MOONEY_SPLIT / $TRIAL_SET / $BLINK_MODE) and return the resolved path dict."""
    return paths_for(ask_mooney_split(), ask_trial_set(), ask_blink_mode())
