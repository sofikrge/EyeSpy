"""Run-mode selection and output paths for the NSS pipeline.

Every script that reads or writes cross-phase outputs calls `select()` at startup. The
modes come from Settings.py, so a full run needs no arguments and no prompts:

    MOONEY_SPLIT  and  TRIAL_SET   read from Settings.py
    BLINK_MODE                     derived from Settings.INTERPOLATE_BLINKS

Deriving the blink mode is what stops the analysis reading a results folder Stage 1 never
built. The three values can still be overridden for one-off runs by setting the
MOONEY_SPLIT / TRIAL_SET / BLINK_MODE environment variables.

Folder layout, relative to the project root ("analysesresults" is Settings.ANALYSES_ROOT,
so a data_rep run writes the same tree under analysesresults_rep/). <suffix> is the trial-set suffix followed by
the blink-mode suffix, e.g. "", "_interp", "_exponly", "_exponly_interp":

    analysesresults/NSS<suffix>/            shared across Mooney modes
        FixMaps_full.pkl
        NSS_WithinPhase.pkl
        NSS_WithinPhase_LongFormat.csv
    analysesresults/NSS_<mode><suffix>/     mode = whole | halves
        NSS_crossphase_descriptives.pkl
        NSS_CrossPhase_LongFormat.csv
        NSS_CrossPhase_LongFormat_centred.csv

Only the cross-phase step depends on the Mooney split, so FixMaps and within-phase are
shared across modes and never rebuilt when you switch. The trial set changes the input
fixations themselves, so it versions everything.
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


def _from_settings(var: str, valid: tuple, settings_attr: str) -> str:
    """Resolve one run mode: the environment wins if set, otherwise Settings.py."""
    env = os.environ.get(var, "").strip().lower()
    if env:
        if env not in valid:
            raise SystemExit(f"${var}={env!r} is not one of {valid}.")
        print(f"[NSSPaths] {var} = {env}  (environment override)")
        return env

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root
    import Settings

    value = getattr(Settings, settings_attr)
    if value not in valid:
        raise SystemExit(f"Settings.{settings_attr} = {value!r} is not one of {valid}.")
    print(f"[NSSPaths] {var} = {value}  (Settings.py)")
    return value


def ask_mooney_split() -> str:
    """Mooney-window mode: "whole" or "halves"."""
    return _from_settings("MOONEY_SPLIT", VALID, "MOONEY_SPLIT")


def ask_trial_set() -> str:
    """Trial set: "all", "experiment" or "extra"."""
    return _from_settings("TRIAL_SET", VALID_TRIAL_SET, "TRIAL_SET")


def ask_blink_mode() -> str:
    """Blink mode, derived from Settings.INTERPOLATE_BLINKS.

    It has to match how Stage 1 built the parquet, and the parquet does not record it.
    Deriving it from the same flag Stage 1 used removes that whole class of mistake: the
    mode both picks the *_interp output folder and gets stamped into NSS.py's cache meta.
    """
    env = os.environ.get("BLINK_MODE", "").strip().lower()
    if env:
        if env not in VALID_BLINK:
            raise SystemExit(f"$BLINK_MODE={env!r} is not one of {VALID_BLINK}.")
        print(f"[NSSPaths] BLINK_MODE = {env}  (environment override)")
        return env

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root
    import Settings

    mode = "interp" if Settings.INTERPOLATE_BLINKS else "filter"
    print(f"[NSSPaths] BLINK_MODE = {mode}  (from Settings.INTERPOLATE_BLINKS={Settings.INTERPOLATE_BLINKS})")
    return mode


def paths_for(mooney_split: str, trial_set: str = "all", blink_mode: str = "filter") -> dict:
    """Resolve every NSS output path for the given mode/trial set/blink mode and ensure the folder exists."""
    if mooney_split not in VALID:
        raise ValueError(f"mooney_split must be one of {VALID}, got {mooney_split!r}")
    if trial_set not in VALID_TRIAL_SET:
        raise ValueError(f"trial_set must be one of {VALID_TRIAL_SET}, got {trial_set!r}")
    if blink_mode not in VALID_BLINK:
        raise ValueError(f"blink_mode must be one of {VALID_BLINK}, got {blink_mode!r}")
    suffix = TRIAL_SET_SUFFIX[trial_set] + BLINK_SUFFIX[blink_mode]  # trial set, then blink mode
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root
    from Settings import ANALYSES_ROOT   # "analysesresults", or "analysesresults_rep"
    shared_dir = Path(f"{ANALYSES_ROOT}/NSS{suffix}")  # mode-independent: FixMaps + within-phase
    out_dir = Path(f"{ANALYSES_ROOT}/NSS_{mooney_split}{suffix}")
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
    """Resolve the run modes from Settings.py (or the env overrides) and return the paths."""
    return paths_for(ask_mooney_split(), ask_trial_set(), ask_blink_mode())
