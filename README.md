# EyeSpy

Eye-tracking analysis pipeline for a Mooney-image / disambiguation consciousness
experiment. It turns raw EyeLink `.asc` recordings (plus MATLAB `.mat` behavioural
files) into cleaned fixation and saccade tables, then runs **NSS (Normalized Scanpath
Saliency)** analyses comparing where participants look during the Mooney phase against
where they looked during the disambiguation phase, split by awareness (PAS score).

This is a collection of analysis scripts rather than an installable package: no CLI,
no test suite.

## Setup

```bash
uv sync
```

Python 3.14. Dependencies are pinned to exact versions in `pyproject.toml`, so `uv sync`
rebuilds the environment these results were produced with. `pymovements` in particular
does the event detection, so upgrading it can change the fixations everything downstream
is built on: bump a version deliberately, then rerun and check the results still hold.

The pipeline expects the data laid out like this (everything under `data/` is
gitignored, only code is tracked):

```
data/my_dataset/raw/s_<SESSION>_<PID>.asc
data/my_dataset/behavioural/expdata_<SESSION>_<PID>.mat
```

`SESSION` is a letter (`C` conscious, `U` unconscious) and `PID` is digits.

## The pipeline

Two stages that communicate only through files on disk, never through imports:

```
data/my_dataset/raw/*.asc  +  behavioural/*.mat
    |   Stage 1: CompleteRun.py
    v
data/events_cleaned/*.csv  +  all_events_cleaned.csv
    |   Stage 2: NSSExporter.py
    v
data/NSS_all_fixations_clean.parquet
    |   NSS.py
    v
analysesresults/    pickled caches + long-format CSVs for Jamovi
    |
    v
Figures/            plots
```

### The whole pipeline

```bash
python3 CompleteRun.py
```

Runs everything: preprocessing, then `NSSExporter.py`, then `NSS.py`. It takes no
arguments and asks no questions, because every setting comes from `Settings.py`.

Stage 1 runs pymovements (load, `pix2deg`, offset correction, `pos2vel`, IVT event
detection) and then this project's own steps: blink and spatial filtering, trial and
phase reconstruction from the `.asc` marker codes, the behavioural merge, and the
exclusions. Stage 2 then builds the fixation parquet and computes the NSS scores.

The individual scripts still run on their own if you only need one stage.

**Blink handling** has two mutually exclusive modes, set by `INTERPOLATE_BLINKS` in
`Settings.py`. `False` drops blink-overlapping events; `True` PCHIP-interpolates gaze
across short blink gaps instead, so a blink-spanning fixation survives as one fixation.
If you change it, rerun Stage 1 and set `BLINK_MODE` to match in Stage 2. The three
scripts in `Scripts/Preprocessing/Diagnostics/` justify the interpolation parameters
from the raw data and can be run on their own.

### Running one stage at a time

```bash
python3 Scripts/Analysis/NSS/NSSExporter.py   # cleaned CSVs -> one fixation parquet
python3 Scripts/Analysis/NSS/NSS.py           # fixation maps, within/cross-phase NSS, Jamovi CSVs
```

Then any of the checks in `Scripts/Analysis/Checks/` and the plots in
`Scripts/Analysis/Plots/`. All of them read the same run modes from `Settings.py`, so
they always agree with the last full run.

### Analysis modes

Results are versioned by three choices, all resolved from `Settings.py`:

| Setting | Values | What it changes |
|---|---|---|
| `MOONEY_SPLIT` | `whole` \| `halves` | Score the 3 s Mooney window as one unit, or as two 1.5 s halves (Early/Late) |
| `TRIAL_SET` | `all` \| `experiment` \| `extra` | Which block's trials enter the analysis |
| `INTERPOLATE_BLINKS` | `True` \| `False` | Stage-1 blink handling. The Stage-2 blink mode is **derived** from it, so the analysis can only read a folder Stage 1 built |

For a one-off run without editing `Settings.py`, the environment overrides them:

```bash
TRIAL_SET=extra python3 Scripts/Analysis/NSS/NSS.py
```

Output folders are suffixed to match: `analysesresults/NSS_whole/`,
`NSS_whole_exponly_interp/`, and so on. `Scripts/Analysis/NSS/NSSPaths.py` is the single
source of truth for these paths.

To compare the two blink-handling methods, set `INTERPOLATE_BLINKS` in `Settings.py`,
run `CompleteRun.py`, then flip it and run again. Each mode writes to its own results
folder, so the two never overwrite each other.

## Repo map

| Path | What it is |
|---|---|
| `Settings.py` | Every tunable parameter for both stages. The one file to edit |
| `CompleteRun.py` | Runs the whole pipeline: preprocessing, export, NSS |
| `Scripts/Preprocessing/` | The Stage-1 steps pymovements does not cover, one file per concern: gaze correction, blink interpolation, event filtering, trial metadata |
| `Scripts/Preprocessing/Diagnostics/` | Standalone justification of the blink-interpolation parameters |
| `Scripts/Analysis/NSS/` | The export, the NSS calculation, and `NSSPaths.py` (run modes and output paths) |
| `Scripts/Analysis/Checks/` | Diagnostics on the cached results: dropped images, cell sizes, per-participant coverage |
| `Scripts/Analysis/Plots/` | Figures |

Every script's docstring says what it reads and what it writes.

## Configuration

- **`Settings.py` is the only file to edit.** Both stages import their parameters from
  it: screen geometry, thresholds, blink handling, marker codes, the participant and
  block exclusions (`EXCLUDE_SESSIONS`, `EXCLUDE_BLOCKS`), and the Stage-2 canvas size,
  pixels-per-degree and subject thresholds.
- That includes the run modes (`MOONEY_SPLIT`, `TRIAL_SET`), which select the results
  folder. The blink mode is not a separate setting: it follows from
  `INTERPOLATE_BLINKS`.
- `DEBUG` gates I/O, not just logging: with it on, the intermediate `data/events/` CSVs
  and all the QC figures under `DataQualityChecks/` get written. `NSS_DEBUG` is the
  separate Stage-2 flag.
- `Scripts/Analysis/Plots/MooneysOnDisamb.py` hardcodes absolute stimulus-image paths
  that must be edited to run on another machine.

## Notes

- Stage 1 uses **polars**, Stage 2 uses **pandas**. The parquet is the handoff.
- `awareness` combines session and PAS response into `conscious_aware`,
  `unconscious_aware` and `unconscious_unaware` (PAS 0 = unaware; PAS 1 is dropped).
- Heavy NSS results are pickle-cached and keyed by a meta dict. If you change a
  parameter that affects them, bump the `tag=` string in `NSS.py` or the cache will
  quietly hand back the old result.
- Methods citations for specific parameters are in [REFERENCES.md](REFERENCES.md).
