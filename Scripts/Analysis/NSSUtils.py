# Scripts/Analysis/NSSUtils.py

"""
Shared utilities for NSS analysis.
Imported by NSSWithinPhase.py and NSSCrossPhase.py.

Contains:
- All configuration constants
- Data loading
- Coordinate conversion helpers
- Cache metadata helpers
- Fixation map building (CreateFixationMaps_from_df)
- All NSS core computation helpers
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import pickle
import zlib
from scipy.ndimage import gaussian_filter

mpl.rcParams.update({
    "savefig.transparent": True,
    "figure.facecolor":    "none",
    "axes.facecolor":      "none",
})

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Set to True  → blend all fixations for participant × image (original behaviour)
# Set to False → keep each trial separate (recommended when images repeat)
# Must match the value used in NSSExporter.py
BLEND_TRIALS = False

FIX_FILE    = Path("data/NSS_all_fixations_clean.parquet")
OUTPUT_DIR  = Path("analysesresults/NSS")
FIGURES_DIR = Path("Figures/nss_analyses")

IMAGE_HEIGHT = 600
IMAGE_WIDTH  = 800
IMAGE_W_DEG  = 9.99
MASK_PPD     = IMAGE_WIDTH / IMAGE_W_DEG   # 80.08 px/deg

DEBUG = True

PALETTE = ['#edf8fb', '#b3cde3', '#8c96c6', '#88419d']

MIN_SUBJ_PER_GROUP_NSS   = 2   # min participants per fixmap group (within-phase)
MIN_SUBJ_PER_GROUP_CROSS = 2   # min participants per fixmap group (cross-phase)
NAN_POLICY_CROSS         = "permissive"   # "permissive" or "matlab_strict"

DISPERSION_DDOF = 0   # 0 = population std; 1 = sample std


# ── GROUP COLUMNS ─────────────────────────────────────────────────────────────

def _group_cols(blend: bool) -> list[str]:
    """
    Return the column names that define one fixation-map unit.
    blend=True  → image × session × image_type                (original)
    blend=False → image × session × image_type × trial_number (per-trial)
    """
    base = ["ImageName", "session", "image_type"]
    return base if blend else base + ["trial_number"]


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_fixations(path: Path = FIX_FILE) -> pd.DataFrame:
    """Load the cleaned fixation parquet written by NSSExporter.py."""
    if not path.exists():
        raise FileNotFoundError(f"Fixations parquet not found: {path}")
    df = pd.read_parquet(path)
    if DEBUG:
        mode = "BLEND" if "trial_number" not in df.columns else "PER-TRIAL"
        print(f"  Loaded {len(df):,} fixation rows [{mode} mode detected from parquet].")
    return df


def validate_parquet_mode(fixations: pd.DataFrame, blend: bool) -> pd.DataFrame:
    """
    Check that the parquet matches the requested blend mode.
    In per-trial mode, overwrite trial_number for disambiguation images
    with 'ALL_TRIALS' so they are pooled into a single reference map.
    Returns the (possibly modified) fixations dataframe.
    """
    parquet_is_pertrial = "trial_number" in fixations.columns

    if blend and parquet_is_pertrial:
        print("  ⚠️  BLEND_TRIALS=True but parquet contains trial_number. "
              "Dropping it to enforce blend mode.")
        fixations = fixations.drop(columns=["trial_number"])

    if not blend and not parquet_is_pertrial:
        raise RuntimeError(
            "BLEND_TRIALS=False but parquet has no trial_number column. "
            "Re-run NSSExporter with BLEND_TRIALS=False."
        )

    if not blend:
        # Pool all disambiguation fixations across trials into one reference map
        # per image × session, by marking them with a sentinel trial label.
        # Mooney fixations keep their real trial_number.
        fixations = fixations.copy()
        fixations["trial_number"] = fixations["trial_number"].astype(object)
        ref_mask = fixations["image_type"].isin(["disamb_intact", "disamb_not_intact"])
        fixations.loc[ref_mask, "trial_number"] = "ALL_TRIALS"

    return fixations


# ── COORDINATE HELPERS ────────────────────────────────────────────────────────

def round_half_away_from_zero(x: np.ndarray) -> np.ndarray:
    """Round to nearest integer, with halves rounded away from zero (matches MATLAB)."""
    x = np.asarray(x, dtype=float)
    return (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(int)


def _deg_to_image_pixels(x_deg, y_deg, ppd, *, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    """Convert centered visual degrees to 1-based pixel coordinates in the image."""
    w = round_half_away_from_zero((width  / 2.0) + x_deg * ppd).astype(int)
    h = round_half_away_from_zero((height / 2.0) + y_deg * ppd).astype(int)
    return h, w   # (row, col), 1-based


# ── CACHE METADATA ────────────────────────────────────────────────────────────

def _meta_block(ppd, image_h, image_w, group_cols, *, tag, extra=None):
    """Build the metadata dict stored alongside each cached result."""
    base = {
        "pixels_per_vdegree": float(ppd),
        "sigma_px":           float(ppd) / 2.0,
        "image_size":         (int(image_h), int(image_w)),
        "group_cols":         tuple(group_cols),
        "blend_trials":       BLEND_TRIALS,
        "format":             tag,
    }
    if extra:
        base.update(extra)
    return base


# ── FIXATION MAPS ─────────────────────────────────────────────────────────────

def CreateFixationMaps_from_df(df: pd.DataFrame, pixels_per_vdegree: float,
                                blend: bool = BLEND_TRIALS) -> list[dict]:
    """
    Build one fixation map entry per group.

    blend=True  → group = (ImageName, session, image_type)
                  inner loop over participants pools all their trials.
    blend=False → group = (ImageName, session, image_type, trial_number)
                  inner loop over participants who saw that exact trial.
                  Disambiguation images use trial_number='ALL_TRIALS' so they
                  are pooled into one stable reference map per image × session.
    """
    needed = {"ImageName", "session", "image_type", "participant", "x_deg_centered", "y_deg"}
    if not blend:
        needed.add("trial_number")
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in fixations DF: {missing}")

    sigma  = float(pixels_per_vdegree) / 2.0
    H, W   = IMAGE_HEIGHT, IMAGE_WIDTH
    ImSize = (H, W)
    gcols  = _group_cols(blend)

    FixMaps = []

    for group_key, df_img in df.sort_values(gcols + ["participant"]).groupby(gcols, dropna=False):

        if blend:
            img_name, cond, img_type = group_key
            trial_num = None
        else:
            img_name, cond, img_type, trial_num = group_key

        entry = {
            "img":          img_name,
            "condition":    cond,
            "image_type":   img_type,
            "trial_number": trial_num,   # None in blend mode
        }

        subj_maps = []
        mapPerIm  = None

        for jj, (pid, df_subj) in enumerate(df_img.groupby("participant", dropna=False), start=1):
            subj_entry      = {"subjNum": jj, "ParticipantID": str(pid)}
            mapPerImAndSubj = np.zeros(ImSize, dtype=np.uint32)

            xdeg = pd.to_numeric(df_subj["x_deg_centered"], errors="coerce").to_numpy()
            ydeg = pd.to_numeric(df_subj["y_deg"],          errors="coerce").to_numpy()
            ok   = np.isfinite(xdeg) & np.isfinite(ydeg)

            if ok.any():
                h1, w1 = _deg_to_image_pixels(xdeg[ok], ydeg[ok], pixels_per_vdegree,
                                               width=W, height=H)
                inside = (h1 >= 1) & (h1 <= H) & (w1 >= 1) & (w1 <= W)
                if inside.any():
                    h1  = h1[inside] - 1
                    w1  = w1[inside] - 1
                    lin = (h1 * W + w1).astype(np.int64)
                    mapPerImAndSubj += np.bincount(lin, minlength=H * W).reshape(ImSize).astype(np.uint16, copy=False)

            if mapPerIm is None:
                mapPerIm = mapPerImAndSubj.astype(np.float32, copy=False)
            else:
                mapPerIm += mapPerImAndSubj

            subj_entry["z"]     = zlib.compress(mapPerImAndSubj.tobytes(), level=1)
            subj_entry["shape"] = ImSize
            del mapPerImAndSubj
            subj_maps.append(subj_entry)

        mapPerIm /= float(max(len(subj_maps), 1))
        gaussian_filter(mapPerIm, sigma=sigma, mode="reflect", truncate=2.0, output=mapPerIm)

        entry["subject"]     = subj_maps
        entry["fixMapPerIm"] = mapPerIm
        FixMaps.append(entry)

    return FixMaps


def get_fixmaps(fixations: pd.DataFrame, ppd: float, blend: bool,
                output_dir: Path) -> list[dict]:
    """
    Load FixMaps from cache if the metadata matches, otherwise build and cache them.
    Called by both run_within_phase_nss and run_cross_phase_nss so the maps
    are only built once per run.
    """
    gcols      = _group_cols(blend)
    cache_path = output_dir / f"FixMaps_{'blend' if blend else 'pertrial'}.pkl"
    meta       = _meta_block(ppd, IMAGE_HEIGHT, IMAGE_WIDTH, gcols,
                             tag="CreateFixationMaps:v3",
                             extra={"ref_blending": "ALL_TRIALS" if not blend else "per_trial"})
    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        if cache.get("meta") == meta:
            FixMaps = cache["data"]
            print(f"  Loaded FixMaps from cache → {cache_path}")
            return FixMaps
        raise ValueError("meta changed")
    except Exception:
        pass

    print("  Building fixation maps…")
    FixMaps = CreateFixationMaps_from_df(fixations, ppd, blend=blend)
    with open(cache_path, "wb") as f:
        pickle.dump({"meta": meta, "data": FixMaps}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved FixMaps → {cache_path}")
    return FixMaps


def summarise_fixmaps(FixMaps: list[dict]) -> pd.DataFrame:
    rows = []
    for e in FixMaps:
        rows.append({
            "img":          e["img"],
            "condition":    e["condition"],
            "image_type":   e["image_type"],
            "trial_number": e.get("trial_number"),
            "n_subjects":   len(e.get("subject", [])),
        })
    df = pd.DataFrame(rows)
    print(f"\n  FixMap summary: {len(df)} groups total")
    print(df["condition"].value_counts().to_string())
    print(df["image_type"].value_counts().to_string())
    print(df["n_subjects"].describe().to_string())
    return df


# ── NSS CORE HELPERS ──────────────────────────────────────────────────────────

def _disk_offsets(radius_px: float):
    r = int(np.ceil(float(radius_px)))
    y, x = np.mgrid[-r:r+1, -r:r+1]
    m = (x*x + y*y) <= float(radius_px)**2
    return y[m].astype(np.int32), x[m].astype(np.int32)


def _disk_means_at_points(zmap, rows_0b, cols_0b, dy, dx, chunk_size=512):
    H, W = zmap.shape
    K    = rows_0b.size
    out  = np.empty(K, dtype=np.float32)
    flat = zmap.ravel()

    for start in range(0, K, chunk_size):
        end = min(start + chunk_size, K)
        r0  = rows_0b[start:end][:, None]
        c0  = cols_0b[start:end][:, None]
        rr  = r0 + dy[None, :]
        cc  = c0 + dx[None, :]
        inb = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)

        if not inb.any():
            out[start:end] = np.nan
            continue

        lin    = rr * W + cc
        vals   = flat[lin[inb]]
        counts = inb.sum(axis=1, dtype=np.int64)
        splits = np.cumsum(counts[:-1], dtype=np.int64)
        parts  = np.split(vals, splits)
        out[start:end] = np.array(
            [p.mean(dtype=np.float64) if p.size else np.nan for p in parts],
            dtype=np.float32,
        )
    return out


def _sigma_from_ppd(ppd: float) -> float:
    return float(ppd) / 2.0


def _z_normalize(arr: np.ndarray):
    mu = float(arr.mean())
    sd = float(arr.std(ddof=DISPERSION_DDOF))
    if sd == 0.0 or not np.isfinite(sd):
        return None, mu, sd
    return (arr - mu) / sd, mu, sd


def _stack_subject_maps(subjects: list[dict], H: int, W: int) -> np.ndarray:
    if not subjects:
        return np.zeros((0, H, W), dtype=np.float32)
    maps = []
    for s in subjects:
        if "mapPerImAndSubj" in s:
            arr = s["mapPerImAndSubj"]
        else:
            arr = np.frombuffer(zlib.decompress(s["z"]), dtype=np.uint32).reshape(s["shape"])
        maps.append(arr)
    out = np.stack(maps, axis=0).astype(np.float32, copy=False)
    del maps
    return out


def _blur_subject_maps(maps: np.ndarray, sigma: float):
    if maps.size == 0:
        return maps, np.zeros(maps.shape[1:], dtype=np.float32)
    blurred  = gaussian_filter(maps, sigma=(0.0, sigma, sigma), mode="reflect", truncate=2.0)
    sum_blur = blurred.sum(axis=0, dtype=np.float32)
    return blurred, sum_blur


def _compute_loso(sum_blur, blurred, j):
    """Leave-one-subject-out average map for subject j."""
    n = blurred.shape[0]
    return (sum_blur - blurred[j]) / float(max(n - 1, 1))


def _nss_for_subject(zmap, coords_j, dy_off, dx_off) -> float:
    if zmap is None or coords_j is None:
        return float("nan")
    h1, w1 = coords_j
    if h1.size == 0:
        return float("nan")
    per_fix = _disk_means_at_points(zmap, h1 - 1, w1 - 1, dy_off, dx_off)
    return float(np.nanmean(per_fix))


def _coords_for_participant(df_subj: pd.DataFrame, ppd: float, H: int, W: int):
    xdeg = pd.to_numeric(df_subj["x_deg_centered"], errors="coerce").to_numpy()
    ydeg = pd.to_numeric(df_subj["y_deg"],          errors="coerce").to_numpy()
    ok   = np.isfinite(xdeg) & np.isfinite(ydeg)
    if not ok.any():
        return np.array([], np.int32), np.array([], np.int32)
    h1, w1 = _deg_to_image_pixels(xdeg[ok], ydeg[ok], ppd, width=W, height=H)
    inside = (h1 >= 1) & (h1 <= H) & (w1 >= 1) & (w1 <= W)
    return np.int32(h1[inside]), np.int32(w1[inside])


def _coords_in_group_order(df_group: pd.DataFrame, ppd: float, H: int, W: int) -> list:
    """
    Return list of (h1,w1) arrays one per participant, sorted by participant ID string.
    Order must match the subject order inside the corresponding FixMap entry.
    """
    coords = []
    dfg = df_group.assign(_pid=df_group["participant"].astype(str)).sort_values("_pid")
    for _, df_subj in dfg.groupby("_pid", dropna=False):
        coords.append(_coords_for_participant(df_subj, ppd, H, W))
    return coords


def _aggregate_by_policy(scores: list[float], policy: str) -> float:
    arr = np.asarray(scores, dtype=float)
    if policy == "matlab_strict":
        return float(arr.mean()) if np.all(np.isfinite(arr)) else float("nan")
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")