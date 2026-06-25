# EyeSpy

Eye-tracking analysis pipeline for a Mooney-image / disambiguation consciousness
experiment. It turns raw EyeLink `.asc` recordings (plus MATLAB `.mat` behavioural
files) into cleaned fixation/saccade event tables, then runs **NSS (Normalized
Scanpath Saliency)** analyses comparing where participants look during the Mooney
phase vs. the disambiguation phase, split by awareness (PAS score).

This is a research script collection, not a packaged library. There is no test
suite and no CLI; all real code lives in `RunningScript.py`, `Settings.py`, and
`Scripts/`.

## Requirements

- Python 3.14 (see `.python-version`)
- Environment managed by [uv](https://docs.astral.sh/uv/) (`.venv/`)
- Core libraries: `pymovements`, `polars` (Stage 1), `pandas`, `numpy`, `scipy`,
  `matplotlib`, `seaborn` (Stage 2)

Run scripts with the virtual environment active, or via `uv run`.

> Note: `pymovements` and `polars` are imported throughout Stage 1 but are not yet
> declared in `pyproject.toml`. See [Known issues](#known-issues).

## Pipeline

The two stages communicate **only through files on disk** under `data/` (gitignored),
not through imports.

```
data/my_dataset/raw/s_<SESSION>_<PID>.asc  +  behavioural/expdata_<SESSION>_<PID>.mat
        |  (Stage 1: RunningScript.py + Scripts/Preprocessing/)
        v
data/events_cleaned/s_<SESSION>_<PID>.csv  +  all_events_cleaned.csv
        |  (Stage 2: NSSExporter.py)
        v
data/NSS_all_fixations_clean.parquet
        |  (NSS.py)
        v
analysesresults/NSS/  (pickled caches + Jamovi CSVs)
        |
        v
Figures/   (CrossNSSViolinPlot.py, MooneysOnDisamb.py, FixationDensityPlot.py)
```

## Running

### Stage 1: preprocessing (raw -> cleaned events)

```bash
python3 RunningScript.py
```

Top-to-bottom orchestration script (`#%%` cells, runnable in an interactive window).
Reads everything from `Settings.py` and writes per-session CSVs to
`data/events_cleaned/` plus a combined `all_events_cleaned.csv`.

### Stage 2: NSS analysis (cleaned events -> scores & figures)

Run in order:

```bash
python3 Scripts/Analysis/NSS/NSSExporter.py        # events_cleaned CSVs -> data/NSS_all_fixations_clean.parquet
python3 Scripts/Analysis/NSS/NSS.py                # fixation maps, within/cross-phase NSS, Jamovi CSVs
python3 Scripts/Analysis/NSS/CrossNSSViolinPlot.py # figure from NSS_CrossPhase_LongFormat.csv
```

### Diagnostic & control scripts

Run after `NSS.py` (see `Scripts/Analysis/NSSControlAnalyses/`):

```bash
python3 Scripts/Analysis/NSSControlAnalyses/NSSInventory.py
python3 Scripts/Analysis/NSSControlAnalyses/MooneysOnDisamb.py
python3 Scripts/Analysis/NSSControlAnalyses/ParticipantsEffectUCUARanking.py
python3 Scripts/Analysis/NSSControlAnalyses/DiagnoseParticipantCoverage.py
python3 Scripts/Analysis/NSSControlAnalyses/FixationDensityPlot.py
```

`MooneysOnDisamb.py` hardcodes absolute stimulus-image paths (`MOONEY_DIRS` /
`DISAMB_DIRS`) that must be edited to run on another machine.

## Configuration

- `Settings.py` is the single source of truth for Stage 1.
- Stage-2 scripts each have their own `=== CONFIG ===` block at the top and do **not**
  import `Settings.py`. Shared constants (image size, PPD) must be kept consistent by
  hand.
- Participant exclusions live in `Settings.py` (`EXCLUDE_SESSIONS`, `EXCLUDE_BLOCKS`).

## Notes

- Filename grammar: `s_<SESSION>_<PID>` where SESSION is a letter (`C` = conscious,
  `U` = unconscious) and PID is digits. `awareness` combines session + PAS response:
  `conscious_aware`, `unconscious_aware`, `unconscious_unaware` (PAS 0 = unaware;
  PAS 1 is dropped).
