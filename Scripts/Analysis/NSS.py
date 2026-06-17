# OldNSS.py

# yes yes yes

"""
NSS (Normalized Scanpath Saliency) Analysis Module for Eye-Gaze Data

This script computes NSS similarity metrics for eye-gaze fixations across different image types
and visibility conditions (conscious vs. unconscious). It performs within-phase and cross-phase
NSS analyses using leave-one-subject-out (LOSO) methodology with Gaussian-blurred fixation maps.
Key Functionality:
- Load and preprocess fixation data with participant/session exclusions
- Convert pixel coordinates to visual degrees and map to image pixel space
- Generate per-subject and group-level fixation maps with Gaussian filtering
- Calculate within-phase NSS: average fixation map saliency for each observer within each experiment phase (disambiguating intact vs scrambled, mooney)
- Calculate cross-phase NSS: Mooney image fixations against disambiguation reference maps (intact vs scrambled)
- Export participant-level results for 2x3 and 2x2 ANOVA designs in Jamovi
- Generate summary statistics, visualizations, and verification ANOVAs
Outputs:
- Cached fixation maps and NSS computations (*.pkl), if changed something and want to rerun script, delete this file first!
- Summary parquets with condition-wise aggregates (*.parquet)
- CSV exports for Jamovi statistical analysis (*.csv)
- Publication-ready plots (*.png)
"""

#%%
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
import pickle
from typing import Any
import matplotlib.pyplot as plt
import zlib
import matplotlib as mpl
import seaborn as sns
from statsmodels.stats.anova import AnovaRM

mpl.rcParams.update({
    "savefig.transparent": True,  # make saved figs have transparent background
    "figure.facecolor": "none",   # transparent figure
    "axes.facecolor": "none"      # transparent axes
})

#%% === CONFIG ===
FIX_FILE            = Path("data/NSS_all_fixations_clean.parquet")
OUTPUT_DIR          = Path("analysesresults/NSS") ; OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR         = Path("Figures/nss_analyses") ; FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SCREEN_WIDTH_PX     = 1920
SCREEN_WIDTH_CM     = 53.2
VIEWING_DIST_CM     = 74.0
IMAGE_HEIGHT        = 600
IMAGE_WIDTH         = 800
DEBUG               = True

PALETTE = ['#edf8fb', '#b3cde3', '#8c96c6', '#88419d']

IMAGE_W_DEG = 9.99
IMAGE_H_DEG = 7.50
MASK_PPD = 48.55  # according to screen dimensions

MIN_SUBJ_PER_IMAGE_NSS   = 2   # within-phase NSS: minimum subjects required per image
MIN_SUBJ_PER_IMAGE_CROSS = 2   # cross-phase NSS: minimum Mooney subjects required per image
# NaN Policy for Cross-Phase NSS:
# - "permissive": Use Intact OR Scrambled reference (whichever exists)
# - "matlab_strict": Require BOTH references (drop image if either missing)
NAN_POLICY_CROSS         = "permissive"  # or "matlab_strict"

DISPERSION_DDOF = 0   # 0 = population sd (spread of present data); set to 1 for sample sd

#%% === LOAD & PREP ===

def load_fixations(path: Path = FIX_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fixations file not found: {path}")
    df = pd.read_parquet(path)
    if DEBUG:
        print(f"Loaded {len(df):,} fixation rows from {path}")
    return df

def pixels_to_vdegrees(pixels, *, screen_width_px=SCREEN_WIDTH_PX,screen_width_cm=SCREEN_WIDTH_CM,viewing_distance_cm=VIEWING_DIST_CM):
    pixels = np.asarray(pixels, dtype=float)
    cm_per_pixel = screen_width_cm / float(screen_width_px)
    return np.degrees(2.0 * np.arctan(((pixels * 0.5) * cm_per_pixel) / float(viewing_distance_cm)))

def _deg_to_image_pixels(x_deg, y_deg, ppd, *, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    w_1based = round_half_away_from_zero((width  / 2.0) + x_deg * ppd).astype(int)
    h_1based = round_half_away_from_zero((height / 2.0) + y_deg * ppd).astype(int)  # ← FIXED: removed minus sign
    return h_1based, w_1based

def round_half_away_from_zero(x):
    x = np.asarray(x, dtype=float)
    return (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(int)

def pixels_per_visual_degree(
    screen_width_px=SCREEN_WIDTH_PX,
    screen_width_cm=SCREEN_WIDTH_CM,
    viewing_distance_cm=VIEWING_DIST_CM,) -> float:
    # CORRECTED: Return mask PPD (800 pixels / 9.99 degrees)
    return IMAGE_WIDTH / 9.99  # = 80.08 pixels per degree

def _meta_block(ppd: float, image_h: int, image_w: int, group_cols: tuple[str, ...], *,
                tag: str, extra: dict | None = None) -> dict:
    base = {
        "pixels_per_vdegree": float(ppd),
        "sigma_px": float(ppd) / 2.0,
        "image_size": (int(image_h), int(image_w)),
        "group_cols": tuple(group_cols),
        "format": tag,
    }
    if extra: base.update(extra)
    return base


#%% === FIXATION MAPS ===
def CreateFixationMaps_from_df(df: pd.DataFrame, pixels_per_vdegree: float):
    """Per-subject hit map → average across subjects → Gaussian blur (MATLAB-equivalent)."""
    needed = {"ImageName", "session", "image_type", "participant", "x_deg_centered", "y_deg"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in fixations DF: {missing}")
    
    sigma  = float(pixels_per_vdegree) / 2.0
    H, W   = IMAGE_HEIGHT, IMAGE_WIDTH
    ImSize = (H, W)

    FixMaps = []
    
    group_cols = ["ImageName", "session", "image_type"]
    for (img_name, cond, img_type), df_img in (
        df.sort_values(group_cols + ["participant"]).groupby(group_cols, dropna=False)
    ):
        entry = {"img": img_name, "condition": cond, "image_type": img_type}

        subj_maps = []
        mapPerIm = None

        for jj, (pid, df_subj) in enumerate(df_img.groupby("participant", dropna=False), start=1):
            subj_entry = {"subjNum": jj, "ParticipantID": str(pid)}
            mapPerImAndSubj = np.zeros(ImSize, dtype=np.uint32)
            xdeg = pd.to_numeric(df_subj.get("x_deg_centered"), errors="coerce").to_numpy()
            ydeg = pd.to_numeric(df_subj.get("y_deg"),          errors="coerce").to_numpy()
            ok = np.isfinite(xdeg) & np.isfinite(ydeg)
            if ok.any():
                h1, w1 = _deg_to_image_pixels(xdeg[ok], ydeg[ok], pixels_per_vdegree,
                                              width=W, height=H)

                inside = (h1 >= 1) & (h1 <= H) & (w1 >= 1) & (w1 <= W)
                if inside.any():
                    h1 = h1[inside] - 1  # 0-based
                    w1 = w1[inside] - 1
                    # ravel to 1D indices and count with bincount (faster)
                    lin = (h1 * W + w1).astype(np.int64)
                    counts = np.bincount(lin, minlength=H * W).reshape(ImSize)
                    mapPerImAndSubj += counts.astype(np.uint16, copy=False)

            # 1) first accumulate into the running average buffer
            if mapPerIm is None:
                mapPerIm = mapPerImAndSubj.astype(np.float32, copy=False)
            else:
                mapPerIm += mapPerImAndSubj

            # 2) then store a compressed copy for the subject and free the temporary
            subj_entry["z"] = zlib.compress(mapPerImAndSubj.tobytes(), level=1)
            subj_entry["shape"] = ImSize
            del mapPerImAndSubj
            subj_maps.append(subj_entry)

        # average across subjects (same as MATLAB's ./size(...,2))
        nsubj = max(len(subj_maps), 1)
        mapPerIm /= float(nsubj)

        # Gaussian blur with symmetric padding (matches MATLAB imgaussfilt 'symmetric')
        gaussian_filter(mapPerIm, sigma=sigma, mode="reflect", truncate=2.0, output=mapPerIm)
        entry["subject"] = subj_maps
        entry["fixMapPerIm"] = mapPerIm

        FixMaps.append(entry)

    return FixMaps

def summarise_fixmaps(FixMaps, n=10):
    """
    Print summary of how many groups per condition / image_type,
    and subjects per group.
    """
    rows = []
    for entry in FixMaps:
        rows.append({
            "img": entry["img"],
            "condition": entry["condition"],
            "image_type": entry["image_type"],
            "n_subjects": len(entry.get("subject", [])),
        })
    df = pd.DataFrame(rows)

    print(f"\nSummary: {len(df)} groups total")

    # Breakdown by condition
    print("\nGroups per condition:")
    print(df["condition"].value_counts())

    # Breakdown by image_type
    print("\nGroups per image_type:")
    print(df["image_type"].value_counts())

    # Subject stats
    print("\nSubjects per group (distribution):")
    print(df["n_subjects"].describe())

    # Example rows
    # print("\nExamples:")
    # print(df.head(n))

    return df

#%% === NSS like Shaked's Matlab ===
def _disk_offsets(radius_px: float) -> tuple[np.ndarray, np.ndarray]:
    r = int(np.ceil(float(radius_px)))   # ← ceil instead of floor
    y, x = np.mgrid[-r:r+1, -r:r+1]
    m = (x*x + y*y) <= (float(radius_px) ** 2)  # still exact inclusion test
    return y[m].astype(np.int32), x[m].astype(np.int32)

def _disk_means_at_points(
    zmap: np.ndarray,
    rows_0b: np.ndarray,
    cols_0b: np.ndarray,
    dy: np.ndarray,
    dx: np.ndarray,
    chunk_size: int = 512
) -> np.ndarray:
    """
    For each fixation (row_0b, col_0b), take the mean of zmap within the disk offsets (dy, dx).
    Pixels leaving the image bounds are ignored.
    """
    H, W = zmap.shape
    K = rows_0b.size
    out = np.empty(K, dtype=np.float32)
    flat = zmap.ravel()

    for start in range(0, K, chunk_size):
        end = min(start + chunk_size, K)
        r0 = rows_0b[start:end][:, None]   # [k,1]
        c0 = cols_0b[start:end][:, None]   # [k,1]

        rr = r0 + dy[None, :]              # [k,m]
        cc = c0 + dx[None, :]              # [k,m]
        inb = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)

        if not inb.any():
            out[start:end] = np.nan
            continue

        lin = rr * W + cc                  # [k,m] linear indices
        vals = flat[lin[inb]]              # stacked in-bounds samples for all k

        # split back per fixation using counts
        counts = inb.sum(axis=1, dtype=np.int64)  # samples per fixation
        splits = np.cumsum(counts[:-1], dtype=np.int64)
        parts = np.split(vals, splits)
        out[start:end] = np.array(
            [p.mean(dtype=np.float64) if p.size else np.nan for p in parts],
            dtype=np.float32
        )
    return out

def _validate_nss_inputs(fixations_df: pd.DataFrame):
    needed = {"ImageName", "session", "image_type", "participant", "x_deg_centered", "y_deg"}
    missing = needed - set(fixations_df.columns)
    if missing:
        raise ValueError(f"Missing columns in fixations DF: {missing}")

def _sigma_from_ppd(pixels_per_vdegree: float) -> float:
    return float(pixels_per_vdegree) / 2.0

def _group_fixations_for_image(fixations_df: pd.DataFrame, img: Any, cond: Any, img_type: Any) -> pd.DataFrame:
    return fixations_df.query(
        "ImageName == @img and session == @cond and image_type == @img_type"
    )

def _coords_in_fixmaps_order(df_group, pixels_per_vdegree, H, W):
    """Return list of (h1,w1) per subject, ordered by participant (as string)."""
    coords = []
    if df_group.empty:
        return coords

    # stable + robust ordering
    dfg = df_group.assign(participant_str=df_group["participant"].astype(str)).sort_values(["participant_str", "trial_number"])
    g = dfg.groupby(["participant_str", "trial_number"], dropna=False)

    for _, df_subj in g:
        xdeg = pd.to_numeric(df_subj["x_deg_centered"], errors="coerce").to_numpy()
        ydeg = pd.to_numeric(df_subj["y_deg"],          errors="coerce").to_numpy()
        ok = np.isfinite(xdeg) & np.isfinite(ydeg)

        if ok.any():
            h1, w1 = _deg_to_image_pixels(
                x_deg=xdeg[ok], y_deg=ydeg[ok], 
                ppd=pixels_per_vdegree, 
                width=W, height=H
            )
            inside = (h1 >= 1) & (h1 <= H) & (w1 >= 1) & (w1 <= W)
            
            # This print tells us exactly how many fixations survive for this participant
            print(f"DEBUG: Participant {df_subj['participant'].iloc} | "
                  f"{ok.sum()} finite, {inside.sum()} inside bounds ({H}x{W}).")
            
            coords.append((np.int32(h1[inside]), np.int32(w1[inside])))
        else:
            print(f"DEBUG: No finite coordinates for participant {df_subj['participant'].iloc}")
            coords.append((np.array([], np.int32), np.array([], np.int32)))

    return coords

def _stack_subject_maps(subjects: list[dict], H: int, W: int) -> np.ndarray:
    """Stack per-subject hit maps into [n, H, W] float32 array (supports compressed)."""
    if not subjects:
        return np.zeros((0, H, W), dtype=np.float32)

    maps = []
    for s in subjects:
        # Backward compatible: accept either raw array or compressed payload
        if "mapPerImAndSubj" in s:
            arr = s["mapPerImAndSubj"]
        else:
            buf = zlib.decompress(s["z"])
            arr = np.frombuffer(buf, dtype=np.uint32).reshape(s["shape"])
        maps.append(arr)

    out = np.stack(maps, axis=0).astype(np.float32, copy=False)
    del maps  # free intermediates
    return out

def _blur_subject_maps(maps: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian blur all subjects at once, return (blurred[n,H,W], sum_blur[H,W])."""
    if maps.size == 0:
        return maps, np.zeros(maps.shape[1:], dtype=np.float32)
    blurred = gaussian_filter(maps, sigma=(0.0, sigma, sigma), mode="reflect", truncate=2.0)
    sum_blur = blurred.sum(axis=0, dtype=np.float32)
    return blurred, sum_blur

def _compute_loso(sum_blur: np.ndarray, blurred: np.ndarray, j: int) -> np.ndarray:
    """Leave-one-subject-out average map for subject j."""
    n = blurred.shape[0]
    return (sum_blur - blurred[j]) / float(max(n - 1, 1))

def _z_normalize(arr: np.ndarray) -> tuple[np.ndarray | None, float, float]:
    """Population z-normalization (ddof=0). Returns (zmap or None if degenerate, mu, sd)."""
    mu = float(arr.mean())
    sd = float(arr.std(ddof=DISPERSION_DDOF))
    if sd == 0.0 or not np.isfinite(sd):
        return None, mu, sd
    return (arr - mu) / sd, mu, sd

def _nss_for_subject(zmap: np.ndarray | None,coords_j: tuple[np.ndarray, np.ndarray] | None,dy_off: np.ndarray,dx_off: np.ndarray,) -> float:
    """Compute NSS for one subject given zmap and that subject's fixation coords."""
    if zmap is None or coords_j is None:
        return float(np.nan)
    h1, w1 = coords_j
    if h1.size == 0:
        return float(np.nan)
    r0, c0 = (h1 - 1), (w1 - 1)  # to 0-based
    per_fix_means = _disk_means_at_points(zmap, r0, c0, dy_off, dx_off)
    return float(np.nanmean(per_fix_means))

def _aggregate_image_scores(subj_scores: list[float]) -> float:
    arr = np.asarray(subj_scores, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")

def _update_global_stats(per_image_means: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(per_image_means, dtype=float)
    if not np.isfinite(arr).any():
        return float("nan"), float("nan"), float("nan")
    mean_ = float(np.nanmean(arr))
    std_  = float(np.nanstd(arr, ddof=1))
    ste_  = float(std_ / np.sqrt(np.isfinite(arr).sum()))
    return mean_, std_, ste_

def calculate_NSS_similarity(FixMaps,fixations_df: pd.DataFrame,pixels_per_vdegree: float,*,min_subj_per_image_nss: int = 2,
    image_height: int = IMAGE_HEIGHT,image_width: int = IMAGE_WIDTH):
    """
    Same MATLAB logic as before, just decomposed into small helpers.
    """
    H, W = int(image_height), int(image_width)
    _validate_nss_inputs(fixations_df)

    sigma = _sigma_from_ppd(pixels_per_vdegree)
    dy_off, dx_off = _disk_offsets(sigma)

    Results = {"image": [], "meanNSSSimilarityPerImage": [],
               "meanNSSSimilarity": np.nan, "stdNSSSimilarity": np.nan, "steNSSSimilarity": np.nan}

    per_image_means = []

    for fm in FixMaps:
        img, cond, img_type = fm["img"], fm["condition"], fm["image_type"]
        subjects = fm.get("subject", [])
        n = len(subjects)

        out = {"img": img, "condition": cond, "image_type": img_type, "subject": []}

        if n >= int(min_subj_per_image_nss):
            maps = _stack_subject_maps(subjects, H, W)
            blurred, sum_blur = _blur_subject_maps(maps, sigma)

            df_group = _group_fixations_for_image(fixations_df, img, cond, img_type)
            # --- SURGICAL ADDITION: Get Participant IDs ---
            participant_ids = sorted(df_group["participant"].astype(str).unique())
            coords_list = _coords_in_fixmaps_order(df_group, pixels_per_vdegree, H, W)

            if img == '1009.jpg' and cond == 'U':
                print(f"DEBUG: 1009/U/Aware. coords_list length: {len(coords_list)}")
                for i, (h, w) in enumerate(coords_list):
                    print(f"  -> Participant index {i} has {len(h)} valid coordinates.")
            
            n_subj = len(coords_list)

            subj_scores = []
            for j in range(n):
                loso = _compute_loso(sum_blur, blurred, j)
                zmap, _, _ = _z_normalize(loso)
                coords_j = coords_list[j] if j < len(coords_list) else (np.array([], np.int32), np.array([], np.int32))
                nss_j = _nss_for_subject(zmap, coords_j, dy_off, dx_off)
                out["subject"].append({
                    "subjNum": subjects[j].get("subjNum", j + 1),
                    "ParticipantID": participant_ids[j], 
                    "NSSSimPerSubj": nss_j
                })
                subj_scores.append(nss_j)

            img_mean = _aggregate_image_scores(subj_scores)
            Results["image"].append(out)
            Results["meanNSSSimilarityPerImage"].append(img_mean)
            per_image_means.append(img_mean)
        else:
            Results["image"].append(out)
            Results["meanNSSSimilarityPerImage"].append(np.nan)
            per_image_means.append(np.nan)

    per_image_means = np.asarray(per_image_means, dtype=np.float32)
    mean_, std_, ste_ = _update_global_stats(per_image_means)
    Results["meanNSSSimilarity"] = mean_
    Results["stdNSSSimilarity"]  = std_
    Results["steNSSSimilarity"]  = ste_

    return Results

def summarise_nss_to_parquet(NSSResults: dict, out_path: Path) -> pd.DataFrame:
    """
    Build a 6-row summary grouped by (condition, image_type) from NSSResults.
    Only includes images whose per-image NSS is finite (like MATLAB's drop-NaN step).
    """
    rows = []
    img_means = NSSResults.get("meanNSSSimilarityPerImage", [])
    images = NSSResults.get("image", [])
    for i, rec in enumerate(images):
        img_mean = img_means[i] if i < len(img_means) else np.nan
        cond = rec.get("condition")
        img_type = rec.get("image_type")
        n_subj = len(rec.get("subject", []))
        rows.append({
            "condition": cond,
            "image_type": img_type,
            "img_mean": float(img_mean),
            "n_subjects": int(n_subj) if np.isfinite(img_mean) else 0,  # count subjects only for kept images
        })

    df = pd.DataFrame(rows)
    kept = df[np.isfinite(df["img_mean"])].copy()
    if kept.empty:
        # still write an empty frame with expected columns
        out = kept.groupby(["condition", "image_type"], dropna=False).size().reset_index(name="n_images_kept")
        out["subjects_total"] = 0
        out["mean_NSS"] = np.nan
        out["std_NSS"] = np.nan
        out["ste_NSS"] = np.nan
    else:
        grp = kept.groupby(["condition", "image_type"], dropna=False)
        out = grp.agg(
            n_images_kept=("img_mean", "count"),
            subjects_total=("n_subjects", "sum"),
            mean_NSS=("img_mean", "mean"),
            std_NSS=("img_mean", lambda x: float(np.std(x.to_numpy(dtype=float), ddof=1)) if len(x) > 1 else np.nan),
        ).reset_index()
        out["ste_NSS"] = out["std_NSS"] / np.sqrt(out["n_images_kept"].clip(lower=1))

    out = out.sort_values(["condition", "image_type"]).reset_index(drop=True)
    out.to_parquet(out_path, index=False)
    print(f"Saved NSS summary → {out_path}")
    return out

def plot_nss_summary(parquet_path: Path):
    """
    Load NSS summary parquet (condition × image_type) and save grouped bar plot.
    - Colors: c = green, u = purple
    - Bars grouped by image_type so c/u are next to each other
    """
    df = pd.read_parquet(parquet_path)

    # Ensure consistent ordering of image types
    image_types = ["disamb_intact", "disamb_not_intact", "mooney_post_intact"]
    conditions = ["C", "U"]
    color_map = {"C": PALETTE[1], "U": PALETTE[0]}

    # Build grouped data
    df = df.set_index(["image_type", "condition"]).sort_index()
    x = range(len(image_types))
    width = 0.35  # bar width per condition

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    for i, cond in enumerate(conditions):
        offset = (i - 0.5) * width
        vals = [df.loc[(img_type, cond), "mean_NSS"] if (img_type, cond) in df.index else np.nan
                for img_type in image_types]
        errs = [df.loc[(img_type, cond), "ste_NSS"] if (img_type, cond) in df.index else np.nan
                for img_type in image_types]
        ax.bar([pos + offset for pos in x], vals, width=width, yerr=errs, capsize=5, alpha=0.8, color=color_map[cond], label=cond)



    ax.set_ylabel("Mean NSS")
    ax.set_title("Within-phase NSS - Image-level means")
    ax.set_xticks(x)
    ax.set_xticklabels(image_types, rotation=30, ha="right")
    ax.legend(title="Condition")

    plt.tight_layout()
    # Save to the new Figures folder, keeping the same filename
    out_path = FIGURES_DIR / parquet_path.with_suffix(".png").name
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved NSS summary plot → {out_path}")

#%% === NSS cross-phase ===
def _index_fixmaps(FixMaps):
    """
    Build a fast lookup: (img, condition, image_type) -> entry
    """
    idx = {}
    for fm in FixMaps:
        idx[(fm["img"], fm["condition"], fm["image_type"])] = fm
    return idx

def _aggregate_by_policy(subj_scores: list[float], policy: str) -> float:
    """
    policy: "permissive" -> np.nanmean; "matlab_strict" -> NaN if any NaN present.
    """
    arr = np.asarray(subj_scores, dtype=float)
    if policy == "matlab_strict":
        return float(arr.mean()) if np.all(np.isfinite(arr)) else float("nan")
    # default permissive
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")

def _summarize_by_condition(per_image_records: list[dict]) -> list[dict]:
    """
    Collapse across images within each condition separately for:
    intact, scrambled, and (intact - scrambled).
    Uses sample std (ddof=1) and STE = std / sqrt(N) on finite images.
    """
    out = []
    df = pd.DataFrame(per_image_records)
    if df.empty:
        return out
    for cond, d in df.groupby("condition", dropna=False):
        row = {"condition": cond}
        for key in ["NSS_intact_img", "NSS_scrambled_img", "NSS_diff_img"]:
            vals = d[key].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            n = int(finite.sum())
            if n == 0:
                mean = std = ste = float("nan")
            else:
                kept = vals[finite]
                mean = float(kept.mean())
                std  = float(kept.std(ddof=1)) if n > 1 else float("nan")
                ste  = float(std / np.sqrt(n)) if n > 1 else float("nan")
            row[f"mean_{key}"] = mean
            row[f"std_{key}"]  = std
            row[f"ste_{key}"]  = ste
            row[f"n_images_{key}"] = n
        out.append(row)
    return out

def calculate_NSS_crossphase(
    FixMaps,
    fixations_df: pd.DataFrame,
    pixels_per_vdegree: float,
    *,
    image_height: int = IMAGE_HEIGHT,
    image_width: int = IMAGE_WIDTH,
    nan_policy: str = "permissive",   # or "matlab_strict"
    min_subj_per_image_cross: int = 2, 
):
    """
    For each Mooney image × condition, compute NSS of Mooney fixations
    against two reference saliency maps from the disambiguation phase:
      - disamb_intact
      - disamb_not_intact (scrambled)
    Reuses existing helpers and mirrors your NSS choices:
      - sigma = pixels_per_vdegree / 2
      - disk radius = sigma (via _disk_offsets)
      - z-normalization with population std (ddof=0)
    """
    H, W = int(image_height), int(image_width)
    _validate_nss_inputs(fixations_df)

    sigma = _sigma_from_ppd(pixels_per_vdegree)
    dy_off, dx_off = _disk_offsets(sigma)

    # Fast lookup for reference maps
    fm_index = _index_fixmaps(FixMaps)

    # Collect per-image results (for summaries) and detailed per-image structures
    per_image_records = []
    Results = {
        "image": [],  # one entry per (img, condition) mooney group, with subject details
        "meanNSS_intact_per_image": [],
        "meanNSS_scrambled_per_image": [],
        "meanNSS_diff_per_image": [],
        "summary_by_condition": [],  # filled at the end
    }

    # Iterate over Mooney groups only
    for fm_mooney in FixMaps:
        img, cond, img_type = fm_mooney["img"], fm_mooney["condition"], fm_mooney["image_type"]
        # Change from "mooney" to your new surgical label
        if img_type != "mooney_post_intact":
            continue

        # Prepare Mooney subject fixation coordinates (ordered like subjects)
        df_group_all = _group_fixations_for_image(fixations_df, img, cond, "mooney_post_intact")
        awareness_groups = df_group_all.groupby("awareness")

        for awareness_val, df_group in awareness_groups:
            participant_ids = sorted(
                (df_group["participant"].astype(str) + "_t" + df_group["trial_number"].astype(str)).unique(),
                key=lambda x: (x.split('_t')[0], int(x.split('_t')[1]))
            )

            # --- ADD THIS DEBUG BLOCK ---
            if img == '1009.jpg' and awareness_val == 'unconscious_aware':
                print(f"DEBUG: Checking data for 1009/U/Aware. Rows in df_group: {len(df_group)}")
                if len(df_group) > 0:
                    print(f"DEBUG: Participants found in df_group: {df_group['participant'].unique()}")
                else:
                    print("DEBUG: df_group is empty. The filter dropped all rows.")
            # ----------------------------
            
            coords_list = _coords_in_fixmaps_order(df_group, pixels_per_vdegree, H, W)

            coords_list = _coords_in_fixmaps_order(df_group, pixels_per_vdegree, H, W)
            n_subj = len(coords_list)

            if n_subj < int(min_subj_per_image_cross):
                Results["image"].append({
                    "img": img, "condition": cond, "image_type": "mooney", # Standardize output type
                    "subject": [],
                    "NSS_intact_img": float("nan"),
                    "NSS_scrambled_img": float("nan"),
                    "NSS_diff_img": float("nan"),
                    "awareness": awareness_val,
                })
                Results["meanNSS_intact_per_image"].append(float("nan"))
                Results["meanNSS_scrambled_per_image"].append(float("nan"))
                Results["meanNSS_diff_per_image"].append(float("nan"))
                per_image_records.append({
                    "img": img, "condition": cond,
                    "NSS_intact_img": float("nan"),
                    "NSS_scrambled_img": float("nan"),
                    "NSS_diff_img": float("nan"),
                    "n_subjects": int(n_subj),
                    "awareness": awareness_val,
                })
                continue
            
            # STRICT SESSION MATCHING: Use 'cond' (current session) to look up the reference
            fm_intact = fm_index.get((img, cond, "disamb_intact"))
            fm_scrambled = fm_index.get((img, cond, "disamb_not_intact"))
            
            # Set helper vars so the debug prints below (lines 640+) don't crash
            intact_session = cond if fm_intact else None
            scrambled_session = cond if fm_scrambled else None

            ref_maps = {
                "intact":    fm_intact["fixMapPerIm"]    if fm_intact    and fm_intact.get("fixMapPerIm", None)    is not None and len(np.shape(fm_intact["fixMapPerIm"])) == 2 else None,
                "scrambled": fm_scrambled["fixMapPerIm"] if fm_scrambled and fm_scrambled.get("fixMapPerIm", None) is not None and len(np.shape(fm_scrambled["fixMapPerIm"])) == 2 else None,
            }

            # Z-normalize references (population std); if std==0 or non-finite => zmap=None (treated as NaN NSS)
            zrefs = {}
            for k, ref in ref_maps.items():
                if ref is None:
                    zrefs[k] = None
                else:
                    zref, mu, sd = _z_normalize(np.asarray(ref, dtype=float))
                    zrefs[k] = zref  # None if degenerate

            # Optional but helpful diagnostics
            if DEBUG:
                if ref_maps["intact"] is None:
                    print(f"[cross] ❌ Missing intact reference for img={img}")
                elif zrefs["intact"] is None:
                    print(f"[cross] ⚠️  Degenerate intact map (zero std) for img={img}")
                
                if ref_maps["scrambled"] is None:
                    print(f"[cross] ❌ Missing scrambled reference for img={img}")
                elif zrefs["scrambled"] is None:
                    print(f"[cross] ⚠️  Degenerate scrambled map (zero std) for img={img}")
                
                # NEW: Show which sessions provided references
                if fm_intact or fm_scrambled:
                    status = []
                    if fm_intact:
                        match = "✓ same" if intact_session == cond else "↔ cross"
                        status.append(f"intact from {intact_session} {match}")
                    if fm_scrambled:
                        match = "✓ same" if scrambled_session == cond else "↔ cross"
                        status.append(f"scrambled from {scrambled_session} {match}")
                    
                    # print(f"[cross-session] img={img}, mooney_sess={cond}, {', '.join(status)}")


            # Per-subject NSS for each reference
            subj_out = []
            subj_scores_intact = []
            subj_scores_scrambled = []

            for j in range(n_subj):
                coords_j = coords_list[j] if j < len(coords_list) else (np.array([], np.int32), np.array([], np.int32))

                nss_intact = _nss_for_subject(zrefs["intact"], coords_j, dy_off, dx_off)
                nss_scram  = _nss_for_subject(zrefs["scrambled"], coords_j, dy_off, dx_off)
                nss_diff   = nss_intact - nss_scram if np.isfinite(nss_intact) and np.isfinite(nss_scram) else float("nan")

                mooney_subjects = fm_mooney.get("subject", [])
                subjnum = mooney_subjects[j].get("subjNum", j + 1) if j < len(mooney_subjects) else (j + 1)

                subj_out.append({
                    "subjNum": subjnum, 
                    "ParticipantID": participant_ids[j],  # <--- ADD THIS
                    "NSS_intact": nss_intact, 
                    "NSS_scrambled": nss_scram, 
                    "NSS_diff": nss_diff,
                    "awareness": awareness_val,
                })

                subj_scores_intact.append(nss_intact)
                subj_scores_scrambled.append(nss_scram)

            # Per-image aggregation (policy-controlled)
            img_nss_intact    = _aggregate_by_policy(subj_scores_intact, nan_policy)
            img_nss_scrambled = _aggregate_by_policy(subj_scores_scrambled, nan_policy)
            img_nss_diff      = img_nss_intact - img_nss_scrambled if np.isfinite(img_nss_intact) and np.isfinite(img_nss_scrambled) else float("nan")

            # Store detailed record and flat record
            Results["image"].append({
                "img": img,
                "condition": cond,
                "image_type": "mooney", # Standardize output type for cross-phase results
                "subject": subj_out,
                "NSS_intact_img": img_nss_intact,
                "NSS_scrambled_img": img_nss_scrambled,
                "NSS_diff_img": img_nss_diff,
                "awareness": awareness_val,
            })

            Results["meanNSS_intact_per_image"].append(img_nss_intact)
            Results["meanNSS_scrambled_per_image"].append(img_nss_scrambled)
            Results["meanNSS_diff_per_image"].append(img_nss_diff)

            per_image_records.append({
                "img": img,
                "condition": cond,
                "NSS_intact_img": img_nss_intact,
                "NSS_scrambled_img": img_nss_scrambled,
                "NSS_diff_img": img_nss_diff,
                "n_subjects": int(n_subj),
                "awareness": awareness_val,
            })

    # Condition-wise summaries
    Results["summary_by_condition"] = _summarize_by_condition(per_image_records)

    return Results

def summarise_nss_crossphase_to_parquet(Cross: dict, out_path: Path) -> pd.DataFrame:
    """
    Build a condition × ref_type summary parquet from cross-phase NSS.
    Input: Cross = output of calculate_NSS_crossphase(...)
    Output parquet columns:
        condition, ref_type, n_images, mean_NSS, std_NSS, ste_NSS
    """
    # flatten per-image
    rows = []
    for rec in Cross.get("image", []):
        rows.append({"img": rec["img"], "condition": rec["condition"], "ref_type": "intact",
                     "NSS_img": float(rec.get("NSS_intact_img", np.nan)),
                     "n_subjects": len(rec.get("subject", []))})
        rows.append({"img": rec["img"], "condition": rec["condition"], "ref_type": "scrambled",
                     "NSS_img": float(rec.get("NSS_scrambled_img", np.nan)),
                     "n_subjects": len(rec.get("subject", []))})
    df = pd.DataFrame(rows)

    if df.empty:
        out = pd.DataFrame({
            "condition": [], "ref_type": [], "n_images": [],
            "mean_NSS": [], "std_NSS": [], "ste_NSS": []
        })
    else:
        kept = df[np.isfinite(df["NSS_img"])].copy()
        grp = kept.groupby(["condition", "ref_type"], dropna=False)
        out = grp.agg(
            n_images=("img", "nunique"),
            mean_NSS=("NSS_img", "mean"),
            std_NSS=("NSS_img", lambda x: float(np.std(x.to_numpy(dtype=float), ddof=1)) if len(x) > 1 else np.nan),
        ).reset_index()
        out["ste_NSS"] = out["std_NSS"] / np.sqrt(out["n_images"].clip(lower=1))

    out = out.sort_values(["condition", "ref_type"]).reset_index(drop=True)
    out_path = Path(out_path)
    out.to_parquet(out_path, index=False)
    print(f"Saved NSS cross-phase summary → {out_path}")
    return out

def plot_nss_crossphase_summary(parquet_path: Path):
    """
    Plot grouped bars per condition with Intact vs Scrambled (error = STE).
    Saves PNG next to the parquet, matching your existing plot style.
    """
    df = pd.read_parquet(parquet_path)

    # --- FIX: Define Data Keys vs. Display Labels ---
    data_conditions = ["C", "U"]           # What is in the dataframe index
    display_labels = ["Conscious", "Unconscious"] # What shows on the plot x-axis
    
    ref_types = ["intact", "scrambled"]
    ref_labels = {"intact": "Intact", "scrambled": "Scrambled"}
    palette_ref = {"intact": PALETTE[3], "scrambled": PALETTE[2]}

    # Pivot for easy plotting
    df = df.set_index(["condition", "ref_type"]).sort_index()

    x = range(len(data_conditions))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, ref in enumerate(ref_types):
        offset = (i - 0.5) * width
        vals = [df.loc[(cond, ref), "mean_NSS"] if (cond, ref) in df.index else np.nan for cond in data_conditions]
        errs = [df.loc[(cond, ref), "ste_NSS"] if (cond, ref) in df.index else np.nan for cond in data_conditions]
        ax.bar([pos + offset for pos in x], vals, width=width, yerr=errs, capsize=5, alpha=0.5, color=palette_ref[ref], label=ref_labels[ref])

    ax.set_ylabel("Mean NSS")
    ax.set_title("Cross-phase NSS: Mooney fixations vs disambiguation maps - Image-level")
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.legend(title="Reference")

    plt.tight_layout()
    out_path = FIGURES_DIR / Path(parquet_path).with_suffix(".png").name
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved NSS cross-phase plot → {out_path}")

#%% === MAIN ===
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # <--- ADD THIS
    
    # Build fixmaps
    fixations = load_fixations()

    ppd = MASK_PPD  # Use mask PPD, not screen PPD
    if DEBUG:
        print(f"Pixels per visual degree: {ppd}")

        print("\n[Inventory] Unique images by image_type × condition:")
        print(
            fixations.groupby(["image_type", "session"])["ImageName"]
                    .nunique()
                    .reset_index(name="n_unique_images")
                    .sort_values(["image_type", "session"])
                    .to_string(index=False)
        )
        print("\n[Inventory] Unique images by image_type (overall):")
        print(
            fixations.groupby("image_type")["ImageName"]
                    .nunique()
                    .reset_index(name="n_unique_images")
                    .sort_values("image_type")
                    .to_string(index=False)
        )

    cache_path = OUTPUT_DIR / "FixMaps_full.pkl"
    meta = _meta_block(ppd, IMAGE_HEIGHT, IMAGE_WIDTH, ("ImageName","condition","image_type"), tag="CreateFixationMaps_from_df:v2")

    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        if cache.get("meta") == meta:
            FixMaps = cache["data"]
            print(f"Loaded FixMaps from cache → {cache_path}")
        else:
            raise ValueError("cache params changed")
    except Exception:
        print("Building fixation maps…")
        FixMaps = CreateFixationMaps_from_df(fixations, ppd)
        with open(cache_path, "wb") as f:
            pickle.dump({"meta": meta, "data": FixMaps}, f)
        print(f"Saved FixMaps → {cache_path}")

    print(f"Ready: {len(FixMaps)} images in FixMaps.")

    # === RAW DATA AUDITOR ===
    print(f"\n[DEBUG] Checking raw participant counts for 1009.jpg / U / unconscious_aware")
    # Access the fixations dataframe (already loaded)
    df_raw = fixations[(fixations['ImageName'] == '1009.jpg') & 
                       (fixations['session'] == 'U') & 
                       (fixations['awareness'] == 'unconscious_aware')]
    
    unique_subs = df_raw['participant'].unique()
    print(f"Total fixation events for 1009/U/Aware: {len(df_raw)}")
    print(f"Total unique participants for 1009/U/Aware: {len(unique_subs)}")
    if len(unique_subs) > 0:
        print(f"Participants found: {unique_subs}")
    else:
        print("No fixations found in the raw dataframe for this condition.")
    # ========================

    if DEBUG: 
        summarise_fixmaps(FixMaps)
        # Build a quick index of what's in FixMaps
        fm_keys = {(fm["img"], fm["condition"], fm["image_type"]) for fm in FixMaps}
        mooney_pairs = sorted({(fm["img"], fm["condition"]) for fm in FixMaps if fm["image_type"] == "mooney_post_intact"})

        missing_scr = [(img, cond) for (img, cond) in mooney_pairs
                    if (img, cond, "disamb_not_intact") not in fm_keys]

        print(f"[probe] Mooney (img,cond) pairs missing scrambled reference: {len(missing_scr)}")
        if missing_scr:
            # For each missing pair, show how many *clean* disamb fixations exist (intact vs scrambled)
            q = (fixations.query("image_type in ['disamb_intact','disamb_not_intact']")
                # --- CHANGE 1: "condition" -> "session" ---
                .groupby(["ImageName", "session", "image_type"], dropna=False)
                .size()
                .rename("n_fix")
                .reset_index())

            pivot = (q.pivot_table(
                                # --- CHANGE 2: "condition" -> "session" ---
                                index=["ImageName", "session"],
                                columns="image_type",
                                values="n_fix",
                                fill_value=0)
                    .sort_index())

            # Print just the missing pairs; if KeyError, it means truly zero rows exist
            try:
                print(pivot.loc[missing_scr])
            except KeyError:
                # Fallback: print a per-pair line if the pivot has no entry at all
                for img, cond in missing_scr:
                    # --- CHANGE 3: "condition" -> "session" (in the filters) ---
                    n_intact = int(q[(q["ImageName"]==img) & (q["session"]==cond) & (q["image_type"]=="disamb_intact")]["n_fix"].sum())
                    n_scram  = int(q[(q["ImageName"]==img) & (q["session"]==cond) & (q["image_type"]=="disamb_not_intact")]["n_fix"].sum())
                    print(f"{img} / {cond}: disamb_intact={n_intact}, disamb_not_intact={n_scram}")
    # --- NSS calculation ---
    nss_cache_path = OUTPUT_DIR / "NSS_WithinPhase.pkl"
    nss_meta = _meta_block(ppd, IMAGE_HEIGHT, IMAGE_WIDTH, ("ImageName","condition","image_type"),
                          tag="calculate_NSS_similarity:v2_disk_at_fix",
                          extra={"min_subj_per_image_nss": int(MIN_SUBJ_PER_IMAGE_NSS)})

    try:
        with open(nss_cache_path, "rb") as f:
            nss_cache = pickle.load(f)

        # Case A: new-format cache with meta+data
        if isinstance(nss_cache, dict) and "meta" in nss_cache and "data" in nss_cache:
            if nss_cache["meta"] == nss_meta:
                NSSResults = nss_cache["data"]
                print(f"Loaded NSSResults from cache → {nss_cache_path}")
            else:
                # helpful diff to explain recompute
                diffs = {k: (nss_cache['meta'].get(k), nss_meta.get(k))
                        for k in sorted(set(nss_cache['meta']) | set(nss_meta))
                        if nss_cache['meta'].get(k) != nss_meta.get(k)}
                print("[NSS cache] meta mismatch → recomputing. Diffs:", diffs)
                raise ValueError("meta mismatch")

        # Case B: legacy file: treat whole object as results
        elif isinstance(nss_cache, dict) and "image" in nss_cache and "meanNSSSimilarityPerImage" in nss_cache:
            NSSResults = nss_cache
            print(f"Loaded legacy NSSResults (no meta) → {nss_cache_path}")

        else:
            print("[NSS cache] Unrecognized format → recomputing.")
            raise ValueError("unrecognized cache format")

    except Exception:
        print("Computing NSS similarity (LOSO, disk radius = sigma)...")
        NSSResults = calculate_NSS_similarity(
            FixMaps=FixMaps,fixations_df=fixations,
            pixels_per_vdegree=ppd,min_subj_per_image_nss=int(MIN_SUBJ_PER_IMAGE_NSS),
            image_height=IMAGE_HEIGHT,image_width=IMAGE_WIDTH)
        with open(nss_cache_path, "wb") as f:
            pickle.dump({"meta": nss_meta, "data": NSSResults}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved NSSResults → {nss_cache_path}")


    kept = [m for m in NSSResults["meanNSSSimilarityPerImage"] if np.isfinite(m)]
    print(f"NSS images kept: {len(kept)} / {len(NSSResults['meanNSSSimilarityPerImage'])}")
    print(f"NSS mean = {NSSResults['meanNSSSimilarity']:.4f}, "
          f"std = {NSSResults['stdNSSSimilarity']:.4f}, "
          f"ste = {NSSResults['steNSSSimilarity']:.4f}")

    # --- Write 6-row summary parquet ---
    nss_summary_path = OUTPUT_DIR / "NSS_withinphase_descriptives.parquet"
    summarise_nss_to_parquet(NSSResults, nss_summary_path)
    
    plot_nss_summary(nss_summary_path)
    
    # --- NSS cross-phase calculation ---
    cross_cache_path = OUTPUT_DIR / "NSS_crossphase_descriptives.pkl"
    cross_meta  = _meta_block(ppd, IMAGE_HEIGHT, IMAGE_WIDTH, ("ImageName","condition"),
                          tag="calculate_NSS_crossphase:v2_cross_session",  # ← Changed version tag
                          extra={"nan_policy": str(NAN_POLICY_CROSS),
                                 "min_subj_per_image_cross": int(MIN_SUBJ_PER_IMAGE_CROSS)})

    try:
        with open(cross_cache_path, "rb") as f:
            cross_cache = pickle.load(f)
        if isinstance(cross_cache, dict) and cross_cache.get("meta") == cross_meta:
            CrossResults = cross_cache["data"]
            print(f"Loaded Cross-phase NSS from cache → {cross_cache_path}")
        else:
            raise ValueError("cross-phase cache meta mismatch")
    except Exception:
        print("Computing Cross-phase NSS (Mooney vs Intact/Scrambled)...")
        CrossResults = calculate_NSS_crossphase(
            FixMaps=FixMaps,
            fixations_df=fixations,
            pixels_per_vdegree=ppd,
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
            nan_policy=str(NAN_POLICY_CROSS),
            min_subj_per_image_cross=int(MIN_SUBJ_PER_IMAGE_CROSS),
        )
        with open(cross_cache_path, "wb") as f:
            pickle.dump({"meta": cross_meta, "data": CrossResults}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved Cross-phase NSS → {cross_cache_path}")

    # === DEBUG INSERTION START ===
    print(f"DEBUG: Inspecting 1009.jpg for U/unconscious_aware...")
    found_1009 = False
    for entry in CrossResults['image']:
        if entry['img'] == '1009.jpg' and entry['condition'] == 'U' and entry.get('awareness') == 'unconscious_aware':
            print(f"Found 1009/U/Aware! Subjects: {len(entry['subject'])}")
            if len(entry['subject']) == 0:
                print(" -> WARNING: The subject list is empty for 1009/U/Aware (The filter discarded it).")
            found_1009 = True
            break # Stop looking once found

    if not found_1009:
        print(" -> 1009.jpg/Unconscious_Aware was never created in the CrossResults dictionary.")
    # === DEBUG INSERTION END ===

    # Cross-phase quick report
    intact_vals    = np.asarray(CrossResults["meanNSS_intact_per_image"], dtype=float)
    scrambled_vals = np.asarray(CrossResults["meanNSS_scrambled_per_image"], dtype=float)
    diff_vals      = np.asarray(CrossResults["meanNSS_diff_per_image"], dtype=float)

    keep_i = np.isfinite(intact_vals)
    keep_s = np.isfinite(scrambled_vals)
    keep_d = np.isfinite(diff_vals)
    print(f"Cross-phase kept images: Intact={keep_i.sum()}, Scrambled={keep_s.sum()}, Diff={keep_d.sum()}")

    if keep_i.any():
        print(f"Cross Intact:   mean={np.nanmean(intact_vals):.4f}")
    if keep_s.any():
        print(f"Cross Scramble: mean={np.nanmean(scrambled_vals):.4f}")
    if keep_d.any():
        print(f"Cross Diff:     mean={np.nanmean(diff_vals):.4f}")

    # --- Cross-phase parquet + plot ---
    cross_summary_path = OUTPUT_DIR / "NSS_crossphase_summary.parquet"
    summarise_nss_crossphase_to_parquet(CrossResults, cross_summary_path)
    plot_nss_crossphase_summary(cross_summary_path)
    
    # ==========================================
    # === DATA RETENTION SUMMARY (DEBUG) ===
    # ==========================================
    print(f"\n📊 Data Retention Summary (Analyzed & Averaged for Jamovi):")
    
    # 1. Within-Phase Stats
    w_imgs = NSSResults.get('image', [])
    w_unique = set(entry['img'] for entry in w_imgs)
    w_c = set(entry['img'] for entry in w_imgs if entry['condition'] == 'C')
    w_u = set(entry['img'] for entry in w_imgs if entry['condition'] == 'U')
    print(f"   Within-Phase: {len(w_unique)} unique images (C={len(w_c)}, U={len(w_u)})")

    # 2. Cross-Phase Stats
    c_imgs = CrossResults.get('image', [])
    c_unique = set(entry['img'] for entry in c_imgs)
    c_c = set(entry['img'] for entry in c_imgs if entry['condition'] == 'C')
    c_u = set(entry['img'] for entry in c_imgs if entry['condition'] == 'U')
    print(f"   Cross-Phase:  {len(c_unique)} unique images (C={len(c_c)}, U={len(c_u)})")
    print("-" * 50)

    # ==========================================
    # === JAMOVI EXPORT: WITHIN-PHASE (2x3) ===
    # ==========================================
    print("\n📦 Creating Within-Phase Dataset for Jamovi (2x3 Design)...")
    
    # 1. Flatten the NSSResults
    w_flat = []
    for img_data in NSSResults['image']:
        # ERROR WAS HERE: We do NOT filter out Mooney anymore. 
        # We want ALL 3 types: mooney, disamb_intact, disamb_not_intact
        
        img_type = img_data['image_type'] 
        session  = img_data['condition']  # "C" or "U"
        image_name = img_data['img']
        
        for subj in img_data['subject']:
            if 'ParticipantID' in subj:
                w_flat.append({
                    'Participant': subj['ParticipantID'],
                    'Image': image_name,
                    'Session': session,
                    'ImageType': img_type,
                    'NSS': subj['NSSSimPerSubj']
                })

    pd.DataFrame(w_flat).to_csv(OUTPUT_DIR / "NSS_WithinPhase_Long.csv", index=False)

    if w_flat:
        # 2. Convert to DataFrame and Aggregate
        df_w_long = pd.DataFrame(w_flat)

        # --- NEW: SAVE LONG FORMAT HERE ---
        long_csv_path = OUTPUT_DIR / "NSS_WithinPhase_LongFormat.csv"
        df_w_long.to_csv(long_csv_path, index=False)
        print(f"✅ Saved Within-Phase LONG Dataset to: {long_csv_path}")
        # ----------------------------------

        df_w_agg = df_w_long.groupby(['Participant', 'Session', 'ImageType'], as_index=False).mean(numeric_only=True)

        # Diagnostic: Find incomplete participants
        counts = df_w_agg.groupby('Participant').size()
        expected_rows = 8  # 2 sessions × 4 phases
        incomplete = counts[counts != expected_rows].index.tolist()
        complete = counts[counts == expected_rows].index.tolist()

        print(f"\n⚠️ Within-Phase: {len(incomplete)} participants excluded due to missing cells:")
        for subj in incomplete:
            subj_data = df_w_agg[df_w_agg['Participant'] == subj]
            found = set(zip(subj_data['Session'], subj_data['ImageType']))
            print(f"   ❌ Participant {subj}: Has {len(found)}/8 cells")

        # 3. Pivot to Wide Format
        # We pivot on BOTH Session and ImageType to get 6 columns
        df_w_wide = df_w_agg.pivot(index='Participant', columns=['Session', 'ImageType'], values='NSS')
        
        # 4. Flatten columns (e.g. NSS_C_mooney, NSS_C_disamb_intact, etc.)
        df_w_wide.columns = [f"NSS_{col[0]}_{col[1]}" for col in df_w_wide.columns]
        df_w_wide = df_w_wide.reset_index()
        
        within_csv_path = OUTPUT_DIR / "NSS_WithinPhase_ParticipantLevel_Wide.csv"
        df_w_wide.to_csv(within_csv_path, index=False)
        print(f"✅ Saved Within-Phase 2x3 Dataset to: {within_csv_path}")
        print("   -> Columns should be (approx):")
        print("      NSS_C_mooney, NSS_C_disamb_intact, NSS_C_disamb_not_intact")
        print("      NSS_U_mooney, NSS_U_disamb_intact, NSS_U_disamb_not_intact")
    else:
        print("⚠️ No Participant IDs found. Did you apply the Step 1 patch?")
    # ==========================================
    # === JAMOVI EXPORT: Cross Phase 2x2 ANOVA FORMAT ===
    # ==========================================
    print("\n📦 Creating Participant-Level Dataset for Jamovi (Wide Format)...")

    # 1. Flatten the nested structure
    # We extract every single subject score from every image
    flat_data = []
    for img_data in CrossResults['image']:
        session = img_data['condition'] # "C" or "U"
        image_name = img_data['img']
        
        for subj in img_data['subject']:
            if 'ParticipantID' in subj: # Only works if you did Step 1
                flat_data.append({
                    'Participant': subj['ParticipantID'].split('_t')[0],
                    'Image': image_name,
                    'Session': session,
                    'NSS_Intact': subj['NSS_intact'],
                    'NSS_Scrambled': subj['NSS_scrambled'],
                    'Awareness': img_data['awareness'],
                    'Trial': subj['ParticipantID'].split('_t')[1],
                })

    # 2. Convert to DataFrame and Aggregate
    # We average across all images for each person-session combo
    df_long = pd.DataFrame(flat_data)


    # --- NEW SURGICAL ADDITION ---
    df_long['Trial'] = pd.to_numeric(df_long['Trial'])
    df_long['Experiment_Half'] = df_long.groupby(['Participant', 'Session'])['Trial'].transform(
        lambda x: np.where(x <= x.median(), 'First_Half', 'Second_Half')
    )

    # --- NEW: SAVE LONG FORMAT HERE ---
    # Note: Cross-phase data is currently "wide" regarding Intact/Scrambled columns. 
    # For LMM, you might want these melted into a single "NSS" column and a "Reference" column.
    
    # Let's melt it fully for you so it is perfectly ready for LMM
    df_long_fully_melted = df_long.melt(
        id_vars=['Participant', 'Image', 'Session', 'Awareness', 'Trial', 'Experiment_Half'],
        value_vars=['NSS_Intact', 'NSS_Scrambled'],
        var_name='ReferenceMap', 
        value_name='NSS'
    )
    # Clean the names (remove "NSS_" prefix from ReferenceMap)
    df_long_fully_melted['ReferenceMap'] = df_long_fully_melted['ReferenceMap'].str.replace('NSS_', '')

    cross_long_path = OUTPUT_DIR / "NSS_CrossPhase_LongFormat.csv"
    df_long_fully_melted.to_csv(cross_long_path, index=False)
    print(f"✅ Saved Cross-Phase LONG Dataset to: {cross_long_path}")
    # ----------------------------------
    
    # --- START DEBUG PRINTS ---
    print("\n--- JAMOVI EXPORT DEBUG ---")
    
    # SCENARIO 1: CrossResults is empty or malformed
    if not CrossResults or 'image' not in CrossResults:
        print("❌ DEBUG: 'CrossResults' dictionary is missing or does not have an 'image' key.")
    elif not CrossResults['image']:
        print("❌ DEBUG: 'CrossResults['image']' list is EMPTY. No cross-phase data was generated.")
        print("   This is the root cause. Check the 'calculate_NSS_crossphase' function and its inputs.")
    else:
        print(f"✅ DEBUG: 'CrossResults['image']' contains {len(CrossResults['image'])} records.")

    # SCENARIO 2: flat_data list is empty after the loop
    if not flat_data:
        print("❌ DEBUG: 'flat_data' list is EMPTY after processing CrossResults.")
        if CrossResults and CrossResults['image']:
            print("   This means the loop over 'CrossResults['image']' ran, but no data was appended.")
            print("   Possible reasons:")
            print("   1. The 'subject' list inside each 'img_data' was empty (e.g., due to MIN_SUBJ_PER_IMAGE_CROSS filter).")
            print("   2. The 'ParticipantID' key was missing from the 'subj' dictionaries.")
            # Let's check the first entry for clues
            first_entry = CrossResults['image'][0]
            if 'subject' not in first_entry:
                 print("      -> Clue: The key 'subject' is missing from the first image record.")
            elif not first_entry['subject']:
                 print("      -> Clue: The 'subject' list in the first image record is empty.")
            elif 'ParticipantID' not in first_entry['subject'][0]:
                 print("      -> Clue: The key 'ParticipantID' is missing from the first subject record.")
                 print(f"         First subject record keys: {list(first_entry['subject'][0].keys())}")
    else:
        print(f"✅ DEBUG: 'flat_data' list contains {len(flat_data)} records.")
    
    # SCENARIO 3: The resulting DataFrame is empty or has no 'Participant' column
    if df_long.empty:
        print("❌ DEBUG: The 'df_long' DataFrame is EMPTY. The script will fail on the next line.")
    else:
        print(f"✅ DEBUG: 'df_long' DataFrame created with {len(df_long)} rows and columns: {df_long.columns.tolist()}")
    print("--- END DEBUG PRINTS ---\n")
    # --- END DEBUG PRINTS ---

    # Group by Participant & Session to get their average performance
    df_agg = df_long.groupby(['Participant', 'Session', 'Awareness'], as_index=False).mean(numeric_only=True)

    # 3. Pivot to Wide Format (Rows=Subjects, Cols=Conditions)
    df_wide = df_agg.pivot(index='Participant', columns=['Session', 'Awareness'], values=['NSS_Intact', 'NSS_Scrambled'])
    
    # 4. Flatten the Multi-Level Columns
    # This turns ('NSS_Intact', 'C') into 'NSS_Intact_C'
    df_wide.columns = [f"{col[0]}_{col[1]}_{col[2]}" for col in df_wide.columns]
    df_wide = df_wide.reset_index()

    # 5. Save
    jamovi_path = OUTPUT_DIR / "NSS_CrossPhase_ParticipantLevel_Wide.csv"
    df_wide.to_csv(jamovi_path, index=False)
    
    print(f"✅ Saved Wide Format Dataset to: {jamovi_path}")
    print("   -> Use this for Repeated Measures ANOVA in Jamovi.")
    print("   -> Columns should be: Participant, NSS_Intact_C, NSS_Scrambled_C, NSS_Intact_U, NSS_Scrambled_U")

    # Add to the end of NSS4Feb.py, before the ANOVA section
    if DEBUG:
        print("\n=== CROSS-SESSION DIAGNOSTIC ===")
        
        valid_intact = sum(1 for img in CrossResults['image'] 
                        if np.isfinite(img['NSS_intact_img']))
        valid_scrambled = sum(1 for img in CrossResults['image'] 
                            if np.isfinite(img['NSS_scrambled_img']))
        total = len(CrossResults['image'])
        
        print(f"Images with valid intact reference: {valid_intact}/{total} ({100*valid_intact/total:.1f}%)")
        print(f"Images with valid scrambled reference: {valid_scrambled}/{total} ({100*valid_scrambled/total:.1f}%)")
        
        # Check for cross-session usage
        cross_session_count = 0
        for img_rec in CrossResults['image']:
            # This requires keeping track during computation - see below
            pass

    # ==========================================
    # === VIOLIN PLOT VISUALIZATION ===
    # ==========================================
    print("\n🎻 Generating Split Violin Plot...")
    try:
     

        # 1. Prepare Data for Plotting
        # We reuse 'df_agg' (the participant-level averages) created in the Jamovi step
        plot_df = df_agg.melt(
            id_vars=['Participant', 'Session', 'Awareness'],
            value_vars=['NSS_Intact', 'NSS_Scrambled'],
            var_name='ReferenceMap',
            value_name='NSS'
        )
        # Clean labels for the legend
        plot_df['ReferenceMap'] = plot_df['ReferenceMap'].str.replace('NSS_', '')

        # 2. Setup the Figure
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")

        # 3. Create Split Violin Plot
        # split=True combines Intact/Scrambled into a single violin per Session
        sns.violinplot(
            data=plot_df, 
            x="Session", 
            y="NSS", 
            hue="ReferenceMap", 
            split=True, 
            inner="quart",  # Draws lines for the quartiles inside the violin
            # Matches the Cross-Phase colors
            palette={"Intact": PALETTE[3], "Scrambled": PALETTE[2]},
            cut=0 # Prevents the violin from extending past the real data range
        )

        # 4. Labels and Title
        plt.title("Double Dissociation: NSS Scores by Visibility & Reference", fontsize=14)
        plt.ylabel("NSS Score (Higher = Better Alignment)", fontsize=12)
        plt.xlabel("Visibility Condition", fontsize=12)
        plt.legend(title="Reference Map")

        # 5. Save
        violin_path = FIGURES_DIR / "NSS_Double_Dissociation_Violin.png"
        plt.savefig(violin_path, dpi=300, bbox_inches='tight')
        print(f"✅ Success! Violin plot saved to: {violin_path}")
        # plt.show() # Uncomment if you want to see it pop up

    except ImportError:
        print("⚠️ plotting libraries missing. Run: pip install seaborn matplotlib")
    except Exception as e:
        print(f"⚠️ Plotting failed: {e}")