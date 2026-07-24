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

Python 3.14. The pipeline expects the data laid out like this (everything under `data/`
is gitignored, only code is tracked):

```
data/my_dataset/raw/s_<SESSION>_<PID>.asc
data/my_dataset/behavioural/expdata_<SESSION>_<PID>.mat
```

`SESSION` is a letter (`C` conscious, `U` unconscious) and `PID` is digits.

## The pipeline

Two stages that communicate only through files on disk, never through imports:

```
data/my_dataset/raw/*.asc  +  behavioural/*.mat
    |   Stage 1: RunPreprocessing.py
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

### Stage 1: preprocessing

```bash
python3 RunPreprocessing.py
```

Runs pymovements (load, `pix2deg`, offset correction, `pos2vel`, IVT event
detection) and then this project's own steps: blink and spatial filtering, trial and
phase reconstruction from the `.asc` marker codes, the behavioural merge, and the
exclusions. Everything is configured in `Settings.py`.

**Blink handling** has two mutually exclusive modes, set by `INTERPOLATE_BLINKS` in
`Settings.py`. `False` drops blink-overlapping events; `True` PCHIP-interpolates gaze
across short blink gaps instead, so a blink-spanning fixation survives as one fixation.
If you change it, rerun Stage 1 and set `BLINK_MODE` to match in Stage 2. The three
scripts in `Scripts/Preprocessing/Diagnostics/` justify the interpolation parameters
from the raw data and can be run on their own.

### Stage 2: NSS analysis

Run in order, from the project root:

```bash
python3 Scripts/Analysis/NSS/NSSExporter.py   # cleaned CSVs -> one fixation parquet
python3 Scripts/Analysis/NSS/NSS.py           # fixation maps, within/cross-phase NSS, Jamovi CSVs
```

Then any of the checks in `Scripts/Analysis/Checks/` and the plots in
`Scripts/Analysis/Plots/`.

### Analysis modes

Results are versioned by three choices. Scripts whose output depends on them **prompt at
startup**, so you cannot accidentally run against the wrong version. Set the environment
variables to skip the prompts:

| Variable | Values | What it changes |
|---|---|---|
| `MOONEY_SPLIT` | `whole` \| `halves` | Score the 3 s Mooney window as one unit, or as two 1.5 s halves (Early/Late) |
| `TRIAL_SET` | `all` \| `experiment` \| `extra` | Which block's trials enter the analysis |
| `BLINK_MODE` | `filter` \| `interp` | **Must match how Stage 1 built the parquet.** It is not recorded in the file |

```bash
MOONEY_SPLIT=whole TRIAL_SET=all BLINK_MODE=filter python3 Scripts/Analysis/NSS/NSS.py
```

Output folders are suffixed to match: `analysesresults/NSS_whole/`,
`NSS_whole_exponly_interp/`, and so on. `Scripts/Analysis/NSS/NSSPaths.py` is the single
source of truth for these paths.

`RunAllInterpolations.py` runs the whole thing end to end once per blink mode.

## Repo map

| Path | What it is |
|---|---|
| `Settings.py` | Every tunable parameter for both stages. The one file to edit |
| `RunPreprocessing.py` | Stage-1 entry point |
| `RunAllInterpolations.py` | Runs the full pipeline once per blink mode |
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
- The per-run choices are the exception: Mooney split, trial set and blink mode are
  picked at runtime, because they select which results folder is written.
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
