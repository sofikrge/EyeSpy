# OldNSS.py

"""
NSS (Normalized Scanpath Saliency) Analysis Module for Eye-Gaze Data
"""

#%%
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
import pickle
from typing import Any
import zlib

#%% === CONFIG ===
FIX_FILE            = Path("data/NSS_all_fixations_clean.parquet")
OUTPUT_DIR          = Path("analysesresults/NSS") ; OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_HEIGHT        = 600
IMAGE_WIDTH         = 800
DEBUG               = True

MASK_PPD            = 48.55 
SIGMA               = MASK_PPD / 2.0 # Pixels per visual degree / 2

MIN_SUBJ_PER_IMAGE_NSS   = 2   # within-phase NSS: minimum subjects required per image
MIN_SUBJ_PER_IMAGE_CROSS = 2   # cross-phase NSS: minimum Mooney subjects required per image
# NaN Policy for Cross-Phase NSS:
# - "permissive": Use Intact OR Scrambled reference (whichever exists)
# - "matlab_strict": Require BOTH references (drop image if either missing)
NAN_POLICY_CROSS         = "permissive"  # or "matlab_strict"

DISPERSION_DDOF = 0   # 0 = population sd (spread of present data); set to 1 for sample sd

#%% === LOAD & PREP ===

def load_fixations(path: Path = FIX_FILE) -> pd.DataFrame:
    """Load fixations from Parquet file."""
    if not path.exists():
        raise FileNotFoundError(f"Fixations file not found: {path}")
    df = pd.read_parquet(path)
    return df

def _deg_to_image_pixels(x_deg, y_deg, ppd, *, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    """Convert visual degrees to image pixel coordinates."""
    w_1based = round_half_away_from_zero((width  / 2.0) + x_deg * ppd).astype(int)
    h_1based = round_half_away_from_zero((height / 2.0) + y_deg * ppd).astype(int) 
    return h_1based, w_1based

def round_half_away_from_zero(x):
    """Round to nearest integer, with ties away from zero (like MATLAB's round)."""
    x = np.asarray(x, dtype=float)
    return (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(int)

def _meta_block(ppd: float, image_h: int, image_w: int, group_cols: tuple[str, ...], *,
                tag: str, extra: dict | None = None) -> dict:
    """Generate metadata dictionary for cache use, to ensure cache is only used if metadata matches """
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
    """Per-subject hit map → average across subjects → Gaussian blur"""
    needed = {"ImageName", "session", "image_type", "participant", "x_deg_centered", "y_deg"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in fixations DF: {missing}")
    
    sigma  = float(pixels_per_vdegree) / 2.0
    H, W   = IMAGE_HEIGHT, IMAGE_WIDTH
    ImSize = (H, W)

    FixMaps = []
    
    group_cols = ["ImageName", "session", "image_type"]

    # Splits dataframe into smaller chunks based on imagename, sesison, imagetype
    for (img_name, cond, img_type), df_img in (
        df.sort_values(group_cols + ["participant"]).groupby(group_cols, dropna=False)
    ):
        if img_type == "mooney_post_scrambled": continue  # skip scrambled Mooney as we won't analyze it directly
        entry = {"img": img_name, "condition": cond, "image_type": img_type}

        subj_maps = []
        mapPerIm = None

        # Build indiviudual maps for each participant
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

            if mapPerIm is None:
                mapPerIm = mapPerImAndSubj.astype(np.float32, copy=False)
            else:
                mapPerIm += mapPerImAndSubj

            # 2) then store a compressed copy for the subject and free the temporary
            subj_entry["z"] = zlib.compress(mapPerImAndSubj.tobytes(), level=1)
            subj_entry["shape"] = ImSize
            del mapPerImAndSubj
            subj_maps.append(subj_entry)

        # average across subjects -> later acts as ref map
        nsubj = max(len(subj_maps), 1)
        mapPerIm /= float(nsubj)

        # Gaussian blur with symmetric padding (matches MATLAB imgaussfilt 'symmetric')
        gaussian_filter(mapPerIm, sigma=sigma, mode="reflect", truncate=2.0, output=mapPerIm)
        entry["subject"] = subj_maps
        entry["fixMapPerIm"] = mapPerIm

        FixMaps.append(entry)

    return FixMaps

#%% === NSS like Shaked's Matlab ===
def _disk_offsets(radius_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Creates circle mask to later identify which fix fall within radius"""
    r = int(np.ceil(float(radius_px)))   # ← ceil instead of floor
    y, x = np.mgrid[-r:r+1, -r:r+1]
    m = (x*x + y*y) <= (float(radius_px) ** 2)  # still exact inclusion test
    return y[m].astype(np.int32), x[m].astype(np.int32)

def _disk_means_at_points(
    zmap: np.ndarray, rows_0b: np.ndarray, cols_0b: np.ndarray, dy: np.ndarray,
    dx: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    """
    Takes disk offset stencil and overlays it onto every fix to capture avg local saleincy score
    """
    H, W = zmap.shape
    K = rows_0b.size
    out = np.empty(K, dtype=np.float32)
    flat = zmap.ravel()

    for start in range(0, K, chunk_size):
        end = min(start + chunk_size, K)
        r0 = rows_0b[start:end][:, None]  
        c0 = cols_0b[start:end][:, None]  

        rr = r0 + dy[None, :]              
        cc = c0 + dx[None, :]             
        inb = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)

        if not inb.any():
            out[start:end] = np.nan
            continue

        lin = rr * W + cc        
        vals = flat[lin[inb]]      

        counts = inb.sum(axis=1, dtype=np.int64)  
        splits = np.cumsum(counts[:-1], dtype=np.int64)
        parts = np.split(vals, splits)
        out[start:end] = np.array(
            [p.mean(dtype=np.float64) if p.size else np.nan for p in parts],
            dtype=np.float32
        )
    return out

def _validate_nss_inputs(fixations_df: pd.DataFrame):
    """Ensure each needed column exists"""
    needed = {"ImageName", "session", "image_type", "participant", "x_deg_centered", "y_deg"}
    missing = needed - set(fixations_df.columns)
    if missing:
        raise ValueError(f"Missing columns in fixations DF: {missing}")

def _group_fixations_for_image(fixations_df: pd.DataFrame, img: Any, cond: Any, img_type: Any) -> pd.DataFrame:
    """Return subset of fixations for a specific image, session, and image type."""
    return fixations_df.query(
        "ImageName == @img and session == @cond and image_type == @img_type")

def _coords_in_fixmaps_order(df_group: pd.DataFrame,pixels_per_vdegree: float,H: int, W: int):
    """Return list of (h1,w1) per subject, ordered by participant (as string)."""
    coords = []
    if df_group.empty:
        return coords

    # order by participant as string to ensure consistent ordering with FixMaps
    dfg = df_group.assign(participant_str=df_group["participant"].astype(str)).sort_values(["participant_str", "trial_number"])
    g = dfg.groupby(["participant_str", "trial_number"], dropna=False)

    for _, df_subj in g:
        xdeg = pd.to_numeric(df_subj["x_deg_centered"], errors="coerce").to_numpy()
        ydeg = pd.to_numeric(df_subj["y_deg"],          errors="coerce").to_numpy()
        ok = np.isfinite(xdeg) & np.isfinite(ydeg)
        # Translate into screen pixel matrices 
        if ok.any():
            h1, w1 = _deg_to_image_pixels(
                x_deg=xdeg[ok], y_deg=ydeg[ok],
                ppd=pixels_per_vdegree, width=W, height=H
            )
            inside = (h1 >= 1) & (h1 <= H) & (w1 >= 1) & (w1 <= W) # double check if anything falls out of bounds
            n_out = np.sum(~inside)
            if n_out > 0 and DEBUG:
                print(f"⚠️ [Boundary Filter] Dropped {n_out} out-of-bounds pixels for image: {df_subj['ImageName'].iloc} (Session {df_subj['session'].iloc})")
            
            coords.append((np.int32(h1[inside]), np.int32(w1[inside])))

        else:
            coords.append((np.array([], np.int32), np.array([], np.int32))) # dummy allocation for dead cells, empty arrays
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

    sigma = SIGMA
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

    sigma = SIGMA
    dy_off, dx_off = _disk_offsets(sigma)

    # Fast lookup for reference maps
    fm_index = _index_fixmaps(FixMaps)

    # Count how many Mooney groups exist for sanity check
    n_mi = sum(1 for fm in FixMaps if fm["image_type"]=="mooney_post_intact")
    n_ms = sum(1 for fm in FixMaps if fm["image_type"]=="mooney_post_scrambled")
    print(f"[cross] Mooney groups found: post_intact={n_mi}, post_scrambled={n_ms}")

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
        print(f"[cross]   total fixations for img={img} cond={cond}: {len(df_group_all)}, unique participants: {df_group_all['participant'].nunique()}, unique trials: {df_group_all['trial_number'].nunique()}")

        if DEBUG:
            print(f"[cross] img={img} cond={cond} | awareness groups: {sorted(df_group_all['awareness'].unique())}")

        for awareness_val, df_group in awareness_groups:
            expected_prefix = "conscious" if cond == "C" else "unconscious"
            if not str(awareness_val).startswith(expected_prefix):
                print(f"[cross] ⚠️ MISMATCH: cond={cond}, awareness={awareness_val} (expected prefix '{expected_prefix}')")
            participant_ids = sorted(
                (df_group["participant"].astype(str) + "_t" + df_group["trial_number"].astype(str)).unique(),
                key=lambda x: (x.split('_t')[0], int(x.split('_t')[1]))
            )
            coords_list = _coords_in_fixmaps_order(df_group, pixels_per_vdegree, H, W)
            print(f"[cross]   coords_list length (n trial-participant units): {len(coords_list)}, non-empty: {sum(1 for h,w in coords_list if len(h)>0)}")
            n_subj= len(coords_list)

            # Debug print for awareness group
            n_fix_raw = len(df_group)
            n_fix_kept = sum(len(h) for h, w in coords_list)
            print(f"[cross]   awareness={awareness_val}: n_subj={n_subj}, fixations raw={n_fix_raw}, kept_in_bounds={n_fix_kept}")

            if n_subj< int(min_subj_per_image_cross):
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
                for k in ['intact', 'scrambled']:
                    if ref_maps[k] is None:
                        print(f"[cross] ❌ ref '{k}' missing for img={img} cond={cond}")
                    elif zrefs[k] is None:
                        print(f"[cross] ⚠️  ref '{k}' degenerate (zero std) for img={img} cond={cond}")
                    else:
                        print(f"[cross] ✓ ref '{k}' ready: mean={ref_maps[k].mean():.4f}, std={ref_maps[k].std():.4f}, nonzero_px={np.count_nonzero(ref_maps[k])}")

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
                if DEBUG:
                    print(f"[cross]     trial_unit j={j} pid={participant_ids[j]}: NSS_intact={nss_intact:.4f}, NSS_scrambled={nss_scram:.4f}")
                subj_scores_scrambled.append(nss_scram)

            # Per-image aggregation (policy-controlled)
            img_nss_intact    = _aggregate_by_policy(subj_scores_intact, nan_policy)
            img_nss_scrambled = _aggregate_by_policy(subj_scores_scrambled, nan_policy)
            img_nss_diff      = img_nss_intact - img_nss_scrambled if np.isfinite(img_nss_intact) and np.isfinite(img_nss_scrambled) else float("nan")

            if DEBUG:
                print(f"[cross]   img_agg img={img} cond={cond} awareness={awareness_val}: intact={img_nss_intact:.4f}, scrambled={img_nss_scrambled:.4f}, diff={img_nss_diff:.4f}")

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

    n_total = len(Results["image"])
    n_valid_i = sum(np.isfinite(r["NSS_intact_img"]) for r in Results["image"])
    n_valid_s = sum(np.isfinite(r["NSS_scrambled_img"]) for r in Results["image"])
    print(f"[cross] DONE: {n_total} (img,cond,awareness) groups → valid intact={n_valid_i}, valid scrambled={n_valid_s}")

    if DEBUG:
        from collections import Counter
        tot = Counter(r["awareness"] for r in Results["image"])
        vi  = Counter(r["awareness"] for r in Results["image"] if np.isfinite(r["NSS_intact_img"]))
        vs  = Counter(r["awareness"] for r in Results["image"] if np.isfinite(r["NSS_scrambled_img"]))
        for aw in tot:
            print(f"[cross] awareness={aw}: total={tot[aw]}, valid_intact={vi.get(aw,0)}, valid_scrambled={vs.get(aw,0)}")

    return Results

#%% === MAIN ===
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # <--- ADD THIS
    
    # Build fixmaps
    fixations = load_fixations()

    ppd = MASK_PPD  # Use mask PPD, not screen PPD

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

    if w_flat:
        # 2. Convert to DataFrame and Aggregate
        df_w_long = pd.DataFrame(w_flat)
        long_csv_path = OUTPUT_DIR / "NSS_WithinPhase_LongFormat.csv"
        df_w_long.to_csv(long_csv_path, index=False)
        print(f"✅ Saved Within-Phase LONG Dataset to: {long_csv_path}")
        
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