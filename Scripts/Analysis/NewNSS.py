# Scripts/Analysis/NSSAnalysis.py

"""
NSS (Normalized Scanpath Saliency) Analysis.

FLAG: BLEND_TRIALS  (must match the value used in NSSExporter.py)
  - True  (original behaviour): unit = participant × image × session.
           LOSO reference map is built from all other participants' fixations
           on the same image in the same session, pooling all their trials.
  - False (per-trial mode, recommended when images repeat):
           unit = participant × image × trial × session.
           LOSO reference map is built from all other participants' fixations
           on the same image in the same session, but only the trial that
           corresponds to the same presentation index.
           The long-format CSV will contain trial_number so you can split
           by experiment half in Jamovi.

Outputs (both modes):
  - Cached FixMaps and NSS results (*.pkl) — delete to force recompute
  - Summary parquets (*.parquet)
  - Long-format CSVs for LMM in Jamovi (NSS_WithinPhase_LongFormat.csv,
    NSS_CrossPhase_LongFormat.csv)
  - Wide-format CSVs for RM-ANOVA in Jamovi
  - Diagnostic plots (*.png)
"""

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import zlib
from scipy.ndimage import gaussian_filter
from statsmodels.stats.anova import AnovaRM

mpl.rcParams.update({
    "savefig.transparent": True,
    "figure.facecolor":    "none",
    "axes.facecolor":      "none",
})

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Must match NSSExporter.py
BLEND_TRIALS = False

FIX_FILE    = Path("data/NSS_all_fixations_clean.parquet")
OUTPUT_DIR  = Path("analysesresults/NSS")
FIGURES_DIR = Path("Figures/nss_analyses")

SCREEN_WIDTH_PX  = 1920
SCREEN_WIDTH_CM  = 53.2
VIEWING_DIST_CM  = 74.0
IMAGE_HEIGHT     = 600
IMAGE_WIDTH      = 800
IMAGE_W_DEG      = 9.99
IMAGE_H_DEG      = 7.50
MASK_PPD         = IMAGE_WIDTH / IMAGE_W_DEG   # 80.08 px/deg

DEBUG = True

PALETTE = ['#edf8fb', '#b3cde3', '#8c96c6', '#88419d']

MIN_SUBJ_PER_GROUP_NSS   = 2   # min participants per fixmap group (within-phase)
MIN_SUBJ_PER_GROUP_CROSS = 2   # min participants per fixmap group (cross-phase)
NAN_POLICY_CROSS         = "permissive"   # "permissive" or "matlab_strict"

EXCLUDE_SUBJECTS = []
EXCLUDE_SESSIONS = {"c": [], "u": []}

DISPERSION_DDOF = 0   # 0 = population std; 1 = sample std

# ── HELPER: which columns define a "group" for fixation map building ──────────

def _group_cols(blend: bool) -> list[str]:
    """
    Return the column names that define one fixation-map unit.
    blend=True  → participant × image × session          (original)
    blend=False → participant × image × session × trial  (per-trial)
    """
    base = ["ImageName", "session", "image_type"]
    return base if blend else base + ["trial_number"]


# ── LOAD & EXCLUSIONS ─────────────────────────────────────────────────────────

def make_exclusion_mask(df: pd.DataFrame) -> np.ndarray:
    m = np.zeros(len(df), dtype=bool)
    if EXCLUDE_SUBJECTS:
        excl = set(map(str, EXCLUDE_SUBJECTS))
        m |= df["participant"].astype(str).isin(excl)
    if EXCLUDE_SESSIONS and any(EXCLUDE_SESSIONS.values()):
        p = df["participant"].astype(str)
        c = df["session"].astype(str).str.lower()
        for key, plist in EXCLUDE_SESSIONS.items():
            ids = set(map(str, plist))
            if ids:
                m |= (c == key.lower()) & p.isin(ids)
    return m


def load_fixations(path: Path = FIX_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fixations parquet not found: {path}")
    df = pd.read_parquet(path)
    mask = make_exclusion_mask(df)
    if mask.any():
        print(f"  Excluded {mask.sum():,} fixation rows via exclusion rules.")
        df = df.loc[~mask].copy()
    if DEBUG:
        mode = "BLEND" if "trial_number" not in df.columns else "PER-TRIAL"
        print(f"  Loaded {len(df):,} fixation rows [{mode} mode detected from parquet].")
    return df


# ── COORDINATE HELPERS ────────────────────────────────────────────────────────

def round_half_away_from_zero(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(int)


def _deg_to_image_pixels(x_deg, y_deg, ppd, *, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    w = round_half_away_from_zero((width  / 2.0) + x_deg * ppd).astype(int)
    h = round_half_away_from_zero((height / 2.0) + y_deg * ppd).astype(int)
    return h, w   # (row, col) in 1-based


# ── CACHE META ────────────────────────────────────────────────────────────────

def _meta_block(ppd, image_h, image_w, group_cols, *, tag, extra=None):
    base = {
        "pixels_per_vdegree": float(ppd),
        "sigma_px":           float(ppd) / 2.0,
        "image_size":         (int(image_h), int(image_w)),
        "group_cols":         tuple(group_cols),
        "blend_trials":       BLEND_TRIALS,
        "excluded_subjects":  tuple(sorted(map(str, EXCLUDE_SUBJECTS))),
        "excluded_sessions":  tuple(sorted(
            (k, tuple(sorted(map(str, EXCLUDE_SESSIONS.get(k, [])))))
            for k in ("c", "u")
        )),
        "format": tag,
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
                  inner loop = participants (original behaviour)
    blend=False → group = (ImageName, session, image_type, trial_number)
                  inner loop = participants who saw that exact trial
                  The entry dict carries 'trial_number' for downstream use.
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

    gcols = _group_cols(blend)

    FixMaps = []

    for group_key, df_img in df.sort_values(gcols + ["participant"]).groupby(gcols, dropna=False):

        # Unpack key depending on blend mode
        if blend:
            img_name, cond, img_type = group_key
            trial_num = None
        else:
            img_name, cond, img_type, trial_num = group_key

        entry = {
            "img":        img_name,
            "condition":  cond,
            "image_type": img_type,
            "trial_number": trial_num,   # None in blend mode
        }

        subj_maps = []
        mapPerIm  = None

        for jj, (pid, df_subj) in enumerate(df_img.groupby("participant", dropna=False), start=1):
            subj_entry = {"subjNum": jj, "ParticipantID": str(pid)}
            mapPerImAndSubj = np.zeros(ImSize, dtype=np.uint32)

            xdeg = pd.to_numeric(df_subj["x_deg_centered"], errors="coerce").to_numpy()
            ydeg = pd.to_numeric(df_subj["y_deg"],          errors="coerce").to_numpy()
            ok   = np.isfinite(xdeg) & np.isfinite(ydeg)

            if ok.any():
                h1, w1 = _deg_to_image_pixels(xdeg[ok], ydeg[ok], pixels_per_vdegree,
                                               width=W, height=H)
                inside = (h1 >= 1) & (h1 <= H) & (w1 >= 1) & (w1 <= W)
                if inside.any():
                    h1 = h1[inside] - 1
                    w1 = w1[inside] - 1
                    lin    = (h1 * W + w1).astype(np.int64)
                    counts = np.bincount(lin, minlength=H * W).reshape(ImSize)
                    mapPerImAndSubj += counts.astype(np.uint16, copy=False)

            if mapPerIm is None:
                mapPerIm = mapPerImAndSubj.astype(np.float32, copy=False)
            else:
                mapPerIm += mapPerImAndSubj

            subj_entry["z"]     = zlib.compress(mapPerImAndSubj.tobytes(), level=1)
            subj_entry["shape"] = ImSize
            del mapPerImAndSubj
            subj_maps.append(subj_entry)

        nsubj     = max(len(subj_maps), 1)
        mapPerIm /= float(nsubj)
        gaussian_filter(mapPerIm, sigma=sigma, mode="reflect", truncate=2.0, output=mapPerIm)

        entry["subject"]     = subj_maps
        entry["fixMapPerIm"] = mapPerIm
        FixMaps.append(entry)

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
    print(f"\nFixMap summary: {len(df)} groups total")
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
    H, W  = zmap.shape
    K     = rows_0b.size
    out   = np.empty(K, dtype=np.float32)
    flat  = zmap.ravel()

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
    Return list of (h1,w1) arrays, one per participant, sorted by participant ID string.
    Used to align with the subject order inside a FixMap entry.
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


# ── WITHIN-PHASE NSS ──────────────────────────────────────────────────────────

def calculate_NSS_similarity(
    FixMaps: list[dict],
    fixations_df: pd.DataFrame,
    pixels_per_vdegree: float,
    *,
    min_subj_per_group: int = MIN_SUBJ_PER_GROUP_NSS,
    blend: bool = BLEND_TRIALS,
    image_height: int = IMAGE_HEIGHT,
    image_width:  int = IMAGE_WIDTH,
) -> dict:
    """
    LOSO within-phase NSS.

    In blend mode:  one NSS score per participant × image × session.
    In per-trial mode: one NSS score per participant × image × trial × session.
    """
    H, W   = int(image_height), int(image_width)
    sigma  = _sigma_from_ppd(pixels_per_vdegree)
    dy_off, dx_off = _disk_offsets(sigma)

    Results = {
        "image": [],
        "meanNSSSimilarityPerImage": [],
        "meanNSSSimilarity": np.nan,
        "stdNSSSimilarity":  np.nan,
        "steNSSSimilarity":  np.nan,
    }
    per_group_means = []

    for fm in FixMaps:
        img       = fm["img"]
        cond      = fm["condition"]
        img_type  = fm["image_type"]
        trial_num = fm.get("trial_number")   # None in blend mode
        subjects  = fm.get("subject", [])
        n         = len(subjects)

        out = {
            "img": img, "condition": cond, "image_type": img_type,
            "trial_number": trial_num, "subject": [],
        }

        if n >= int(min_subj_per_group):
            maps     = _stack_subject_maps(subjects, H, W)
            blurred, sum_blur = _blur_subject_maps(maps, sigma)

            # Fetch the fixation rows for this group
            if blend:
                df_group = fixations_df.query(
                    "ImageName == @img and session == @cond and image_type == @img_type"
                )
            else:
                df_group = fixations_df.query(
                    "ImageName == @img and session == @cond "
                    "and image_type == @img_type and trial_number == @trial_num"
                )

            participant_ids = sorted(df_group["participant"].astype(str).unique())
            coords_list     = _coords_in_group_order(df_group, pixels_per_vdegree, H, W)

            subj_scores = []
            for j in range(n):
                loso       = _compute_loso(sum_blur, blurred, j)
                zmap, _, _ = _z_normalize(loso)
                coords_j   = coords_list[j] if j < len(coords_list) \
                             else (np.array([], np.int32), np.array([], np.int32))
                nss_j = _nss_for_subject(zmap, coords_j, dy_off, dx_off)
                out["subject"].append({
                    "subjNum":      subjects[j].get("subjNum", j + 1),
                    "ParticipantID": participant_ids[j] if j < len(participant_ids) else "",
                    "NSSSimPerSubj": nss_j,
                })
                subj_scores.append(nss_j)

            arr      = np.asarray(subj_scores, dtype=float)
            img_mean = float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")
        else:
            img_mean = float("nan")

        Results["image"].append(out)
        Results["meanNSSSimilarityPerImage"].append(img_mean)
        per_group_means.append(img_mean)

    arr   = np.asarray(per_group_means, dtype=float)
    fin   = np.isfinite(arr)
    Results["meanNSSSimilarity"] = float(np.nanmean(arr))  if fin.any() else float("nan")
    Results["stdNSSSimilarity"]  = float(np.nanstd(arr, ddof=1)) if fin.sum() > 1 else float("nan")
    Results["steNSSSimilarity"]  = Results["stdNSSSimilarity"] / np.sqrt(fin.sum()) \
                                   if fin.sum() > 1 else float("nan")
    return Results


# ── CROSS-PHASE NSS ───────────────────────────────────────────────────────────

def _index_fixmaps(FixMaps: list[dict]) -> dict:
    """
    Build lookup: (img, condition, image_type [, trial_number]) → entry.
    trial_number key is included only in per-trial mode (when it is not None).
    """
    idx = {}
    for fm in FixMaps:
        t = fm.get("trial_number")
        if t is None:
            key = (fm["img"], fm["condition"], fm["image_type"])
        else:
            key = (fm["img"], fm["condition"], fm["image_type"], t)
        idx[key] = fm
    return idx


def _summarize_crossphase_by_condition(records: list[dict]) -> list[dict]:
    out = []
    df  = pd.DataFrame(records)
    if df.empty:
        return out
    for cond, d in df.groupby("condition", dropna=False):
        row = {"condition": cond}
        for key in ["NSS_intact_img", "NSS_scrambled_img", "NSS_diff_img"]:
            vals   = d[key].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            n      = int(finite.sum())
            if n == 0:
                row[f"mean_{key}"] = row[f"std_{key}"] = row[f"ste_{key}"] = float("nan")
                row[f"n_groups_{key}"] = 0
            else:
                kept = vals[finite]
                mean = float(kept.mean())
                std  = float(kept.std(ddof=1)) if n > 1 else float("nan")
                ste  = std / np.sqrt(n)         if n > 1 else float("nan")
                row[f"mean_{key}"] = mean
                row[f"std_{key}"]  = std
                row[f"ste_{key}"]  = ste
                row[f"n_groups_{key}"] = n
        out.append(row)
    return out


def calculate_NSS_crossphase(
    FixMaps: list[dict],
    fixations_df: pd.DataFrame,
    pixels_per_vdegree: float,
    *,
    blend: bool = BLEND_TRIALS,
    image_height: int = IMAGE_HEIGHT,
    image_width:  int = IMAGE_WIDTH,
    nan_policy: str = NAN_POLICY_CROSS,
    min_subj_per_group: int = MIN_SUBJ_PER_GROUP_CROSS,
) -> dict:
    """
    Cross-phase NSS: Mooney fixations scored against disambiguation reference maps.

    In blend mode:
        Reference map for (img, session, disamb_intact) is looked up by
        (img, session, disamb_intact) — same as original.

    In per-trial mode:
        The mooney trial has a trial_number. We look up the disambiguation
        reference map for the SAME trial_number, because the participant saw
        the disambiguation image in that same trial before the mooney.
        This ensures we compare mooney gaze to that participant's OWN prior
        disambiguation fixations from the exact same trial, not a different one.
    """
    H, W   = int(image_height), int(image_width)
    sigma  = _sigma_from_ppd(pixels_per_vdegree)
    dy_off, dx_off = _disk_offsets(sigma)
    fm_index = _index_fixmaps(FixMaps)

    per_group_records = []
    Results = {
        "image": [],
        "meanNSS_intact_per_image":    [],
        "meanNSS_scrambled_per_image": [],
        "meanNSS_diff_per_image":      [],
        "summary_by_condition":        [],
    }

    for fm_mooney in FixMaps:
        img       = fm_mooney["img"]
        cond      = fm_mooney["condition"]
        img_type  = fm_mooney["image_type"]
        trial_num = fm_mooney.get("trial_number")

        if img_type != "mooney_post_intact":
            continue

        # Fetch mooney fixation rows for this group
        if blend:
            df_group = fixations_df.query(
                "ImageName == @img and session == @cond and image_type == 'mooney_post_intact'"
            )
        else:
            df_group = fixations_df.query(
                "ImageName == @img and session == @cond "
                "and image_type == 'mooney_post_intact' and trial_number == @trial_num"
            )

        participant_ids = sorted(df_group["participant"].astype(str).unique())
        coords_list     = _coords_in_group_order(df_group, pixels_per_vdegree, H, W)
        n_subj          = len(coords_list)

        nan_entry = {
            "img": img, "condition": cond, "image_type": "mooney_post_intact",
            "trial_number": trial_num, "subject": [],
            "NSS_intact_img": float("nan"),
            "NSS_scrambled_img": float("nan"),
            "NSS_diff_img": float("nan"),
        }

        if n_subj < int(min_subj_per_group):
            Results["image"].append(nan_entry)
            Results["meanNSS_intact_per_image"].append(float("nan"))
            Results["meanNSS_scrambled_per_image"].append(float("nan"))
            Results["meanNSS_diff_per_image"].append(float("nan"))
            per_group_records.append({
                "img": img, "condition": cond, "trial_number": trial_num,
                "NSS_intact_img": float("nan"), "NSS_scrambled_img": float("nan"),
                "NSS_diff_img": float("nan"), "n_subjects": n_subj,
            })
            continue

        # Look up disambiguation reference maps
        # In per-trial mode, look up by the same trial_number so we match
        # the pre-mooney disambiguation exposure from the exact same trial.
        if blend:
            fm_intact    = fm_index.get((img, cond, "disamb_intact"))
            fm_scrambled = fm_index.get((img, cond, "disamb_not_intact"))
        else:
            fm_intact    = fm_index.get((img, cond, "disamb_intact",     trial_num))
            fm_scrambled = fm_index.get((img, cond, "disamb_not_intact", trial_num))

        def _safe_ref(fm):
            if fm is None:
                return None
            ref = fm.get("fixMapPerIm")
            if ref is None or np.ndim(ref) != 2:
                return None
            return ref

        ref_maps = {"intact": _safe_ref(fm_intact), "scrambled": _safe_ref(fm_scrambled)}

        zrefs = {}
        for k, ref in ref_maps.items():
            if ref is None:
                zrefs[k] = None
                if DEBUG:
                    label = "intact" if k == "intact" else "scrambled"
                    t_str = f" trial={trial_num}" if trial_num is not None else ""
                    print(f"  [cross] ❌ Missing {label} reference  img={img}{t_str}")
            else:
                zref, _, _ = _z_normalize(np.asarray(ref, dtype=float))
                zrefs[k] = zref

        subj_out             = []
        subj_scores_intact   = []
        subj_scores_scrambled = []

        for j in range(n_subj):
            coords_j  = coords_list[j] if j < len(coords_list) \
                        else (np.array([], np.int32), np.array([], np.int32))
            nss_i  = _nss_for_subject(zrefs["intact"],    coords_j, dy_off, dx_off)
            nss_s  = _nss_for_subject(zrefs["scrambled"], coords_j, dy_off, dx_off)
            nss_d  = nss_i - nss_s if np.isfinite(nss_i) and np.isfinite(nss_s) else float("nan")

            mooney_subjects = fm_mooney.get("subject", [])
            subjnum = mooney_subjects[j].get("subjNum", j + 1) if j < len(mooney_subjects) else j + 1

            subj_out.append({
                "subjNum":       subjnum,
                "ParticipantID": participant_ids[j] if j < len(participant_ids) else "",
                "NSS_intact":    nss_i,
                "NSS_scrambled": nss_s,
                "NSS_diff":      nss_d,
            })
            subj_scores_intact.append(nss_i)
            subj_scores_scrambled.append(nss_s)

        img_nss_i = _aggregate_by_policy(subj_scores_intact,    nan_policy)
        img_nss_s = _aggregate_by_policy(subj_scores_scrambled, nan_policy)
        img_nss_d = img_nss_i - img_nss_s \
                    if np.isfinite(img_nss_i) and np.isfinite(img_nss_s) else float("nan")

        Results["image"].append({
            "img": img, "condition": cond, "image_type": "mooney_post_intact",
            "trial_number": trial_num, "subject": subj_out,
            "NSS_intact_img": img_nss_i, "NSS_scrambled_img": img_nss_s, "NSS_diff_img": img_nss_d,
        })
        Results["meanNSS_intact_per_image"].append(img_nss_i)
        Results["meanNSS_scrambled_per_image"].append(img_nss_s)
        Results["meanNSS_diff_per_image"].append(img_nss_d)
        per_group_records.append({
            "img": img, "condition": cond, "trial_number": trial_num,
            "NSS_intact_img": img_nss_i, "NSS_scrambled_img": img_nss_s, "NSS_diff_img": img_nss_d,
            "n_subjects": n_subj,
        })

    Results["summary_by_condition"] = _summarize_crossphase_by_condition(per_group_records)
    return Results


# ── SUMMARY / PLOT HELPERS ────────────────────────────────────────────────────

def summarise_nss_to_parquet(NSSResults: dict, out_path: Path) -> pd.DataFrame:
    rows      = []
    img_means = NSSResults.get("meanNSSSimilarityPerImage", [])
    for i, rec in enumerate(NSSResults.get("image", [])):
        img_mean = img_means[i] if i < len(img_means) else np.nan
        rows.append({
            "condition":    rec.get("condition"),
            "image_type":   rec.get("image_type"),
            "trial_number": rec.get("trial_number"),
            "img_mean":     float(img_mean),
            "n_subjects":   len(rec.get("subject", [])) if np.isfinite(img_mean) else 0,
        })
    df   = pd.DataFrame(rows)
    kept = df[np.isfinite(df["img_mean"])].copy()

    if kept.empty:
        out = pd.DataFrame(columns=["condition","image_type","n_groups_kept",
                                     "subjects_total","mean_NSS","std_NSS","ste_NSS"])
    else:
        grp = kept.groupby(["condition", "image_type"], dropna=False)
        out = grp.agg(
            n_groups_kept  =("img_mean", "count"),
            subjects_total =("n_subjects", "sum"),
            mean_NSS       =("img_mean", "mean"),
            std_NSS        =("img_mean", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else np.nan),
        ).reset_index()
        out["ste_NSS"] = out["std_NSS"] / np.sqrt(out["n_groups_kept"].clip(lower=1))

    out = out.sort_values(["condition", "image_type"]).reset_index(drop=True)
    out.to_parquet(out_path, index=False)
    print(f"  Saved within-phase NSS summary → {out_path}")
    return out


def summarise_nss_crossphase_to_parquet(Cross: dict, out_path: Path) -> pd.DataFrame:
    rows = []
    for rec in Cross.get("image", []):
        for ref_type, key in [("intact", "NSS_intact_img"), ("scrambled", "NSS_scrambled_img")]:
            rows.append({
                "img":          rec["img"],
                "condition":    rec["condition"],
                "trial_number": rec.get("trial_number"),
                "ref_type":     ref_type,
                "NSS_img":      float(rec.get(key, np.nan)),
                "n_subjects":   len(rec.get("subject", [])),
            })
    df = pd.DataFrame(rows)

    if df.empty:
        out = pd.DataFrame(columns=["condition","ref_type","n_groups","mean_NSS","std_NSS","ste_NSS"])
    else:
        kept = df[np.isfinite(df["NSS_img"])].copy()
        grp  = kept.groupby(["condition", "ref_type"], dropna=False)
        out  = grp.agg(
            n_groups=("img", "nunique"),
            mean_NSS=("NSS_img", "mean"),
            std_NSS =("NSS_img", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else np.nan),
        ).reset_index()
        out["ste_NSS"] = out["std_NSS"] / np.sqrt(out["n_groups"].clip(lower=1))

    out = out.sort_values(["condition","ref_type"]).reset_index(drop=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"  Saved cross-phase NSS summary → {out_path}")
    return out


def plot_nss_summary(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    image_types = ["disamb_intact", "disamb_not_intact", "mooney_post_intact"]
    conditions  = ["C", "U"]
    color_map   = {"C": PALETTE[1], "U": PALETTE[0]}
    df = df.set_index(["image_type", "condition"]).sort_index()
    x, width = range(len(image_types)), 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.spines[["top","right"]].set_visible(False)
    for i, cond in enumerate(conditions):
        offset = (i - 0.5) * width
        vals = [df.loc[(t, cond), "mean_NSS"] if (t, cond) in df.index else np.nan for t in image_types]
        errs = [df.loc[(t, cond), "ste_NSS"]  if (t, cond) in df.index else np.nan for t in image_types]
        ax.bar([p + offset for p in x], vals, width=width, yerr=errs, capsize=5,
               alpha=0.8, color=color_map[cond], label=cond)
    ax.set_ylabel("Mean NSS")
    ax.set_title("Within-phase NSS")
    ax.set_xticks(x); ax.set_xticklabels(image_types, rotation=30, ha="right")
    ax.legend(title="Session")
    plt.tight_layout()
    out = FIGURES_DIR / Path(parquet_path).with_suffix(".png").name
    plt.savefig(out, dpi=300); plt.close(fig)
    print(f"  Saved plot → {out}")


def plot_nss_crossphase_summary(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    data_conds    = ["C", "U"]
    display_labels = ["Conscious", "Unconscious"]
    palette_ref   = {"intact": PALETTE[3], "scrambled": PALETTE[2]}
    df = df.set_index(["condition","ref_type"]).sort_index()
    x, width = range(len(data_conds)), 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.spines[["top","right"]].set_visible(False)
    for i, ref in enumerate(["intact","scrambled"]):
        offset = (i - 0.5) * width
        vals = [df.loc[(c, ref), "mean_NSS"] if (c, ref) in df.index else np.nan for c in data_conds]
        errs = [df.loc[(c, ref), "ste_NSS"]  if (c, ref) in df.index else np.nan for c in data_conds]
        ax.bar([p + offset for p in x], vals, width=width, yerr=errs, capsize=5,
               alpha=0.5, color=palette_ref[ref], label=ref.capitalize())
    ax.set_ylabel("Mean NSS")
    ax.set_title("Cross-phase NSS: Mooney vs Disambiguation")
    ax.set_xticks(x); ax.set_xticklabels(display_labels)
    ax.legend(title="Reference")
    plt.tight_layout()
    out = FIGURES_DIR / Path(parquet_path).with_suffix(".png").name
    plt.savefig(out, dpi=300); plt.close(fig)
    print(f"  Saved plot → {out}")


# ── JAMOVI EXPORT HELPERS ─────────────────────────────────────────────────────

def _flatten_within_phase(NSSResults: dict) -> pd.DataFrame:
    rows = []
    for img_data in NSSResults["image"]:
        for subj in img_data["subject"]:
            if "ParticipantID" not in subj:
                continue
            rows.append({
                "Participant":  subj["ParticipantID"],
                "Image":        img_data["img"],
                "Session":      img_data["condition"],
                "ImageType":    img_data["image_type"],
                "trial_number": img_data.get("trial_number"),   # None in blend mode
                "NSS":          subj["NSSSimPerSubj"],
            })
    return pd.DataFrame(rows)


def _flatten_cross_phase(CrossResults: dict) -> pd.DataFrame:
    rows = []
    for img_data in CrossResults["image"]:
        for subj in img_data["subject"]:
            if "ParticipantID" not in subj:
                continue
            rows.append({
                "Participant":  subj["ParticipantID"],
                "Image":        img_data["img"],
                "Session":      img_data["condition"],
                "trial_number": img_data.get("trial_number"),   # None in blend mode
                "NSS_Intact":   subj["NSS_intact"],
                "NSS_Scrambled": subj["NSS_scrambled"],
            })
    return pd.DataFrame(rows)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_nss_analysis(blend: bool = BLEND_TRIALS):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    mode_label = "BLEND" if blend else "PER-TRIAL"
    print(f"\n{'='*60}")
    print(f"  NSS Analysis  [{mode_label} mode]")
    print(f"{'='*60}")

    fixations = load_fixations()

    # Detect mode from parquet content (trial_number present = per-trial)
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

    ppd   = MASK_PPD
    gcols = _group_cols(blend)

    if DEBUG:
        print(f"  PPD = {ppd:.4f}   Group cols = {gcols}")
        print(fixations.groupby(["image_type","session"])["ImageName"]
              .nunique().reset_index(name="n_images").to_string(index=False))

    # ── FixMaps cache ─────────────────────────────────────────────────────────
    cache_path = OUTPUT_DIR / f"FixMaps_{'blend' if blend else 'pertrial'}.pkl"
    meta       = _meta_block(ppd, IMAGE_HEIGHT, IMAGE_WIDTH, gcols,
                             tag="CreateFixationMaps:v3")
    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        if cache.get("meta") == meta:
            FixMaps = cache["data"]
            print(f"  Loaded FixMaps from cache → {cache_path}")
        else:
            raise ValueError("meta changed")
    except Exception:
        print("  Building fixation maps…")
        FixMaps = CreateFixationMaps_from_df(fixations, ppd, blend=blend)
        with open(cache_path, "wb") as f:
            pickle.dump({"meta": meta, "data": FixMaps}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved FixMaps → {cache_path}")

    print(f"  FixMaps ready: {len(FixMaps)} groups.")
    if DEBUG:
        summarise_fixmaps(FixMaps)

    # ── Within-phase NSS cache ────────────────────────────────────────────────
    nss_cache_path = OUTPUT_DIR / f"NSS_WithinPhase_{'blend' if blend else 'pertrial'}.pkl"
    nss_meta = _meta_block(ppd, IMAGE_HEIGHT, IMAGE_WIDTH, gcols,
                           tag="calculate_NSS_similarity:v3",
                           extra={"min_subj_per_group": MIN_SUBJ_PER_GROUP_NSS})
    try:
        with open(nss_cache_path, "rb") as f:
            nc = pickle.load(f)
        if isinstance(nc, dict) and nc.get("meta") == nss_meta:
            NSSResults = nc["data"]
            print(f"  Loaded within-phase NSS from cache → {nss_cache_path}")
        else:
            raise ValueError("meta changed")
    except Exception:
        print("  Computing within-phase NSS…")
        NSSResults = calculate_NSS_similarity(
            FixMaps, fixations, ppd,
            min_subj_per_group=MIN_SUBJ_PER_GROUP_NSS, blend=blend,
        )
        with open(nss_cache_path, "wb") as f:
            pickle.dump({"meta": nss_meta, "data": NSSResults}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved within-phase NSS → {nss_cache_path}")

    kept = [m for m in NSSResults["meanNSSSimilarityPerImage"] if np.isfinite(m)]
    print(f"  Within-phase NSS: {len(kept)} / {len(NSSResults['meanNSSSimilarityPerImage'])} groups kept")
    print(f"  mean={NSSResults['meanNSSSimilarity']:.4f}  "
          f"std={NSSResults['stdNSSSimilarity']:.4f}  "
          f"ste={NSSResults['steNSSSimilarity']:.4f}")

    nss_summary_path = OUTPUT_DIR / "NSS_withinphase_descriptives.parquet"
    summarise_nss_to_parquet(NSSResults, nss_summary_path)
    plot_nss_summary(nss_summary_path)

    # ── Cross-phase NSS cache ─────────────────────────────────────────────────
    cross_cache_path = OUTPUT_DIR / f"NSS_CrossPhase_{'blend' if blend else 'pertrial'}.pkl"
    cross_meta = _meta_block(ppd, IMAGE_HEIGHT, IMAGE_WIDTH, gcols,
                             tag="calculate_NSS_crossphase:v3",
                             extra={"nan_policy": NAN_POLICY_CROSS,
                                    "min_subj_per_group": MIN_SUBJ_PER_GROUP_CROSS})
    try:
        with open(cross_cache_path, "rb") as f:
            cc = pickle.load(f)
        if isinstance(cc, dict) and cc.get("meta") == cross_meta:
            CrossResults = cc["data"]
            print(f"  Loaded cross-phase NSS from cache → {cross_cache_path}")
        else:
            raise ValueError("meta changed")
    except Exception:
        print("  Computing cross-phase NSS…")
        CrossResults = calculate_NSS_crossphase(
            FixMaps, fixations, ppd, blend=blend,
            nan_policy=NAN_POLICY_CROSS,
            min_subj_per_group=MIN_SUBJ_PER_GROUP_CROSS,
        )
        with open(cross_cache_path, "wb") as f:
            pickle.dump({"meta": cross_meta, "data": CrossResults}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved cross-phase NSS → {cross_cache_path}")

    for label, key in [("Intact", "meanNSS_intact_per_image"),
                        ("Scrambled", "meanNSS_scrambled_per_image"),
                        ("Diff", "meanNSS_diff_per_image")]:
        arr = np.asarray(CrossResults[key], dtype=float)
        n   = np.isfinite(arr).sum()
        print(f"  Cross-phase {label}: {n} groups kept, mean={np.nanmean(arr):.4f}")

    cross_summary_path = OUTPUT_DIR / "NSS_crossphase_summary.parquet"
    summarise_nss_crossphase_to_parquet(CrossResults, cross_summary_path)
    plot_nss_crossphase_summary(cross_summary_path)

    # ── Jamovi exports ────────────────────────────────────────────────────────
    print("\n  Exporting long-format CSVs for Jamovi LMM…")

    # Within-phase long format
    df_w = _flatten_within_phase(NSSResults)
    if blend:
        # In blend mode there is no trial_number column — drop it
        df_w = df_w.drop(columns=["trial_number"], errors="ignore")
    df_w.to_csv(OUTPUT_DIR / "NSS_WithinPhase_LongFormat.csv", index=False)
    print(f"  ✅ Within-phase long format → {OUTPUT_DIR}/NSS_WithinPhase_LongFormat.csv")

    # Within-phase wide format (participant-level averages, for RM-ANOVA)
    df_w_agg  = df_w.groupby(["Participant","Session","ImageType"], as_index=False)["NSS"].mean()
    df_w_wide = df_w_agg.pivot(index="Participant", columns=["Session","ImageType"], values="NSS")
    df_w_wide.columns = [f"NSS_{c[0]}_{c[1]}" for c in df_w_wide.columns]
    df_w_wide.reset_index().to_csv(OUTPUT_DIR / "NSS_WithinPhase_Wide.csv", index=False)
    print(f"  ✅ Within-phase wide format  → {OUTPUT_DIR}/NSS_WithinPhase_Wide.csv")

    # Cross-phase long format (fully melted, one NSS value per row)
    df_c = _flatten_cross_phase(CrossResults)
    if blend:
        df_c = df_c.drop(columns=["trial_number"], errors="ignore")
    id_vars = ["Participant","Image","Session"] + \
              (["trial_number"] if "trial_number" in df_c.columns else [])
    df_c_melted = df_c.melt(
        id_vars=id_vars,
        value_vars=["NSS_Intact","NSS_Scrambled"],
        var_name="ReferenceMap",
        value_name="NSS",
    )
    df_c_melted["ReferenceMap"] = df_c_melted["ReferenceMap"].str.replace("NSS_", "")
    df_c_melted.to_csv(OUTPUT_DIR / "NSS_CrossPhase_LongFormat.csv", index=False)
    print(f"  ✅ Cross-phase long format   → {OUTPUT_DIR}/NSS_CrossPhase_LongFormat.csv")

    # Cross-phase wide format (participant-level averages, for RM-ANOVA)
    df_c_agg  = df_c.groupby(["Participant","Session"], as_index=False)[["NSS_Intact","NSS_Scrambled"]].mean()
    df_c_wide = df_c_agg.pivot(index="Participant", columns="Session",
                                values=["NSS_Intact","NSS_Scrambled"])
    df_c_wide.columns = [f"{c[0]}_{c[1]}" for c in df_c_wide.columns]
    df_c_wide.reset_index().to_csv(OUTPUT_DIR / "NSS_CrossPhase_Wide.csv", index=False)
    print(f"  ✅ Cross-phase wide format   → {OUTPUT_DIR}/NSS_CrossPhase_Wide.csv")

    # ── Verification ANOVA (cross-phase) ─────────────────────────────────────
    print("\n  Running verification RM-ANOVA (cross-phase)…")
    try:
        df_anova = df_c_melted.copy()
        counts   = df_anova.groupby("Participant").size()
        complete = counts[counts == 4].index
        df_clean = df_anova[df_anova["Participant"].isin(complete)]
        if len(df_clean) == 0:
            print("  ⚠️  No complete participants for ANOVA.")
        else:
            res = AnovaRM(df_clean, depvar="NSS", subject="Participant",
                          within=["Session","ReferenceMap"]).fit()
            print(res)
    except Exception as e:
        print(f"  ⚠️  ANOVA failed: {e}")

    # ── Violin plot ───────────────────────────────────────────────────────────
    print("\n  Generating violin plot…")
    try:
        plot_df = df_c_agg.melt(id_vars=["Participant","Session"],
                                 value_vars=["NSS_Intact","NSS_Scrambled"],
                                 var_name="ReferenceMap", value_name="NSS")
        plot_df["ReferenceMap"] = plot_df["ReferenceMap"].str.replace("NSS_","")
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        sns.violinplot(data=plot_df, x="Session", y="NSS", hue="ReferenceMap",
                       split=True, inner="quart",
                       palette={"Intact": PALETTE[3], "Scrambled": PALETTE[2]}, cut=0)
        plt.title("Cross-phase NSS by Session and Reference Map")
        plt.ylabel("NSS Score"); plt.xlabel("Session")
        plt.legend(title="Reference Map")
        vpath = FIGURES_DIR / "NSS_CrossPhase_Violin.png"
        plt.savefig(vpath, dpi=300, bbox_inches="tight"); plt.close()
        print(f"  ✅ Violin plot → {vpath}")
    except Exception as e:
        print(f"  ⚠️  Violin plot failed: {e}")