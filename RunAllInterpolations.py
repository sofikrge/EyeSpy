"""Run the full pipeline twice: once with blink interpolation, once with blink filtering.

For each pass we (1) set INTERPOLATE_BLINKS in Settings.py accordingly, then run
Stage 1 (RunningScript.py) -> NSSExporter.py -> NSS.py with the matching Stage-2
settings: MOONEY_SPLIT=whole, TRIAL_SET=all, BLINK_MODE=interp|filter.

Run from the project root:  python3 RunAllInterpolations.py
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SETTINGS = ROOT / "Settings.py"

# (label, INTERPOLATE_BLINKS value, BLINK_MODE for Stage 2)
PASSES = [
    ("blink interpolation", True, "interp"),
   #  ("blink filtering", False, "filter"),
]

# Scripts to run in order, after Settings.py is set for the pass.
STAGE1 = ROOT / "RunningScript.py"
NSS_SCRIPTS = [
    ROOT / "Scripts/Analysis/NSS/NSSExporter.py",
    ROOT / "Scripts/Analysis/NSS/NSS.py",
]


def set_interpolate_blinks(value: bool) -> None:
    """Rewrite the `INTERPOLATE_BLINKS = ...` line in Settings.py."""
    text = SETTINGS.read_text()
    new_text, n = re.subn(
        r"^INTERPOLATE_BLINKS\s*=.*$",
        f"INTERPOLATE_BLINKS = {value}",
        text,
        flags=re.MULTILINE,
    )
    if n != 1:
        sys.exit(f"Expected exactly one INTERPOLATE_BLINKS line in Settings.py, found {n}")
    SETTINGS.write_text(new_text)


def run(script: Path, env: dict) -> None:
    print(f"\n=== {script.relative_to(ROOT)} ===", flush=True)
    subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env, check=True)


for label, interpolate, blink_mode in PASSES:
    print(f"\n########## PASS: {label} (INTERPOLATE_BLINKS={interpolate}) ##########")
    set_interpolate_blinks(interpolate)

    # Stage 1 reads INTERPOLATE_BLINKS from Settings.py directly.
    run(STAGE1, os.environ.copy())

    # Stage 2 reads its mode from env vars (whole / all-trials / matching blink mode).
    env = {**os.environ, "MOONEY_SPLIT": "whole", "TRIAL_SET": "all", "BLINK_MODE": blink_mode}
    for script in NSS_SCRIPTS:
        run(script, env)

print("\nBoth passes complete.")
