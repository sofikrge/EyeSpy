# Scripts/Analysis/NSSWithinPhase.py

"""
Within-phase NSS analysis.

For each image × session (blend mode) or image × session × trial (per-trial mode),
computes leave-one-subject-out NSS: each participant's fixations are scored against
the average fixation map of all other participants on the same image.

Outputs:
  - analysesresults/NSS/NSS_withinphase_descriptives.parquet  (summary)
  - Figures/nss_analyses/NSS_withinphase_descriptives.png     (bar plot)
  - analysesresults/NSS/NSS_WithinPhase_LongFormat.csv        (for Jamovi LMM)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from Scripts.Analysis.NSSUtils import (
    BLEND_TRIALS, OUTPUT_DIR, FIGURES_DIR, MASK_PPD,
    IMAGE_HEIGHT, IMAGE_WIDTH, MIN_SUBJ_PER_GROUP_NSS, DEBUG, PALETTE,
    _group_cols, _meta_block, _sigma_from_ppd, _disk_offsets,
    _stack_subject_maps, _blur_subject_maps, _compute_loso, _z_normalize,
    _nss_for_subject, _coords_in_group_order,
    load_fixations, validate_parquet_mode, get_fixmaps, summarise_fixmaps,
)


# ── WITHIN-PHASE NSS COMPUTATION ─────────────────────────────────────────────

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

    blend=True:  one NSS score per participant × image × session.
    blend=False: one NSS score per participant × image × trial × session.
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
        trial_num = fm.get("trial_number")
        subjects  = fm.get("subject", [])
        n         = len(subjects)

        out = {
            "img": img, "condition": cond, "image_type": img_type,
            "trial_number": trial_num, "subject": [],
        }

        if n >= int(min_subj_per_group):
            maps              = _stack_subject_maps(subjects, H, W)
            blurred, sum_blur = _blur_subject_maps(maps, sigma)

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
                    "subjNum":       subjects[j].get("subjNum", j + 1),
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

    arr = np.asarray(per_group_means, dtype=float)
    fin = np.isfinite(arr)
    Results["meanNSSSimilarity"] = float(np.nanmean(arr))       if fin.any()      else float("nan")
    Results["stdNSSSimilarity"]  = float(np.nanstd(arr, ddof=1)) if fin.sum() > 1 else float("nan")
    Results["steNSSSimilarity"]  = Results["stdNSSSimilarity"] / np.sqrt(fin.sum()) \
                                   if fin.sum() > 1 else float("nan")
    return Results


# ── SUMMARY & PLOT ────────────────────────────────────────────────────────────

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
        out = pd.DataFrame(columns=["condition", "image_type", "n_groups_kept",
                                     "subjects_total", "mean_NSS", "std_NSS", "ste_NSS"])
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


def plot_nss_summary(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    image_types = ["disamb_intact", "disamb_not_intact", "mooney_post_intact"]
    conditions  = ["C", "U"]
    color_map   = {"C": PALETTE[1], "U": PALETTE[0]}
    df = df.set_index(["image_type", "condition"]).sort_index()
    x, width = range(len(image_types)), 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.spines[["top", "right"]].set_visible(False)
    for i, cond in enumerate(conditions):
        offset = (i - 0.5) * width
        vals = [df.loc[(t, cond), "mean_NSS"] if (t, cond) in df.index else np.nan for t in image_types]
        errs = [df.loc[(t, cond), "ste_NSS"]  if (t, cond) in df.index else np.nan for t in image_types]
        ax.bar([p + offset for p in x], vals, width=width, yerr=errs, capsize=5,
               alpha=0.8, color=color_map[cond], label=cond)
    ax.set_ylabel("Mean NSS")
    ax.set_title("Within-phase NSS")
    ax.set_xticks(x)
    ax.set_xticklabels(image_types, rotation=30, ha="right")
    ax.legend(title="Session")
    plt.tight_layout()
    out = FIGURES_DIR / Path(parquet_path).with_suffix(".png").name
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved plot → {out}")


# ── JAMOVI EXPORT ─────────────────────────────────────────────────────────────

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
                "trial_number": img_data.get("trial_number"),
                "NSS":          subj["NSSSimPerSubj"],
            })
    return pd.DataFrame(rows)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_within_phase_nss(blend: bool = BLEND_TRIALS):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    mode_label = "BLEND" if blend else "PER-TRIAL"
    print(f"\n{'='*60}")
    print(f"  Within-Phase NSS  [{mode_label} mode]")
    print(f"{'='*60}")

    fixations = load_fixations()
    fixations = validate_parquet_mode(fixations, blend)

    ppd   = MASK_PPD
    gcols = _group_cols(blend)

    if DEBUG:
        print(f"  PPD = {ppd:.4f}   Group cols = {gcols}")
        print(fixations.groupby(["image_type", "session"])["ImageName"]
              .nunique().reset_index(name="n_images").to_string(index=False))

    FixMaps = get_fixmaps(fixations, ppd, blend, OUTPUT_DIR)
    print(f"  FixMaps ready: {len(FixMaps)} groups.")
    if DEBUG:
        summarise_fixmaps(FixMaps)

    # ── Cache ─────────────────────────────────────────────────────────────────
    cache_path = OUTPUT_DIR / f"NSS_WithinPhase_{'blend' if blend else 'pertrial'}.pkl"
    meta = _meta_block(ppd, IMAGE_HEIGHT, IMAGE_WIDTH, gcols,
                       tag="calculate_NSS_similarity:v3",
                       extra={"min_subj_per_group": MIN_SUBJ_PER_GROUP_NSS,
                              "ref_blending": "ALL_TRIALS" if not blend else "per_trial"})
    try:
        with open(cache_path, "rb") as f:
            nc = pickle.load(f)
        if isinstance(nc, dict) and nc.get("meta") == meta:
            NSSResults = nc["data"]
            print(f"  Loaded within-phase NSS from cache → {cache_path}")
        else:
            raise ValueError("meta changed")
    except Exception:
        print("  Computing within-phase NSS…")
        NSSResults = calculate_NSS_similarity(
            FixMaps, fixations, ppd,
            min_subj_per_group=MIN_SUBJ_PER_GROUP_NSS, blend=blend,
        )
        with open(cache_path, "wb") as f:
            pickle.dump({"meta": meta, "data": NSSResults}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved within-phase NSS → {cache_path}")

    kept = [m for m in NSSResults["meanNSSSimilarityPerImage"] if np.isfinite(m)]
    print(f"  Groups kept: {len(kept)} / {len(NSSResults['meanNSSSimilarityPerImage'])}")
    print(f"  mean={NSSResults['meanNSSSimilarity']:.4f}  "
          f"std={NSSResults['stdNSSSimilarity']:.4f}  "
          f"ste={NSSResults['steNSSSimilarity']:.4f}")

    # ── Summary parquet + plot ─────────────────────────────────────────────────
    summary_path = OUTPUT_DIR / "NSS_withinphase_descriptives.parquet"
    summarise_nss_to_parquet(NSSResults, summary_path)
    plot_nss_summary(summary_path)

    # ── Long-format CSV for Jamovi LMM ─────────────────────────────────────────
    df_w = _flatten_within_phase(NSSResults)
    if blend:
        df_w = df_w.drop(columns=["trial_number"], errors="ignore")
    # --- ADD AWARENESS ---
    aw = fixations.rename(columns={"participant": "Participant", "ImageName": "Image", "session": "Session"})
    m_cols = ["Participant", "Image", "Session"] + (["trial_number"] if "trial_number" in df_w.columns else [])
    df_w = df_w.merge(aw[m_cols + ["awareness"]].drop_duplicates(), on=m_cols, how="left")
    df_w.to_csv(OUTPUT_DIR / "NSS_WithinPhase_LongFormat.csv", index=False)
    print(f"  ✅ Within-phase long format → {OUTPUT_DIR}/NSS_WithinPhase_LongFormat.csv")