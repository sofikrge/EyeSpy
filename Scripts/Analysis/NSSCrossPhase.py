
# new
# Scripts/Analysis/NSSCrossPhase.py

"""
Cross-phase NSS analysis.

For each Mooney image × session (blend mode) or × session × trial (per-trial mode),
scores each participant's Mooney fixations against two reference saliency maps built
from the disambiguation phase: one for intact images and one for scrambled images.

Reference map lookup:
  blend=True:  reference maps are looked up by (image, session, disamb_type).
  blend=False: disambiguation fixations are pooled across all trials per image × session
               (trial_number = 'ALL_TRIALS'), so the lookup key is
               (image, session, disamb_type, 'ALL_TRIALS').
               Mooney maps keep their real trial_number.

Outputs:
  - analysesresults/NSS/NSS_crossphase_summary.parquet   (summary)
  - Figures/nss_analyses/NSS_crossphase_summary.png      (bar plot)
  - analysesresults/NSS/NSS_CrossPhase_LongFormat.csv    (for Jamovi LMM)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from Scripts.Analysis.NSSUtils import (
    BLEND_TRIALS, OUTPUT_DIR, FIGURES_DIR, MASK_PPD,
    IMAGE_HEIGHT, IMAGE_WIDTH, MIN_SUBJ_PER_GROUP_CROSS, NAN_POLICY_CROSS,
    DEBUG, PALETTE,
    _group_cols, _meta_block, _sigma_from_ppd, _disk_offsets,
    _stack_subject_maps, _blur_subject_maps, _z_normalize,
    _nss_for_subject, _coords_in_group_order, _aggregate_by_policy,
    load_fixations, validate_parquet_mode, get_fixmaps, summarise_fixmaps,
)


# ── CROSS-PHASE NSS COMPUTATION ───────────────────────────────────────────────

def _index_fixmaps(FixMaps: list[dict]) -> dict:
    """
    Build a fast lookup: key → FixMap entry.

    blend mode key:    (img, condition, image_type)
    per-trial mode key: (img, condition, image_type, trial_number)
    """
    idx = {}
    for fm in FixMaps:
        t = fm.get("trial_number")
        key = (fm["img"], fm["condition"], fm["image_type"]) if t is None \
              else (fm["img"], fm["condition"], fm["image_type"], t)
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
                row[f"mean_{key}"]    = mean
                row[f"std_{key}"]     = std
                row[f"ste_{key}"]     = ste
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
    Cross-phase NSS: score Mooney fixations against disambiguation reference maps.

    blend=True:
        Reference maps looked up as (img, session, disamb_type).
    blend=False:
        Disambiguation maps were built with trial_number='ALL_TRIALS' (set in
        validate_parquet_mode), so the lookup key is
        (img, session, disamb_type, 'ALL_TRIALS').
        Mooney trial_number is kept per-trial.
    """
    H, W   = int(image_height), int(image_width)
    sigma  = _sigma_from_ppd(pixels_per_vdegree)
    dy_off, dx_off = _disk_offsets(sigma)
    fm_index = _index_fixmaps(FixMaps)

    per_group_records = []
    Results = {
        "image":                       [],
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

        # Shared nan result for this group
        def _append_nan():
            Results["image"].append({
                "img": img, "condition": cond, "image_type": "mooney_post_intact",
                "trial_number": trial_num, "subject": [],
                "NSS_intact_img": float("nan"), "NSS_scrambled_img": float("nan"),
                "NSS_diff_img": float("nan"),
            })
            Results["meanNSS_intact_per_image"].append(float("nan"))
            Results["meanNSS_scrambled_per_image"].append(float("nan"))
            Results["meanNSS_diff_per_image"].append(float("nan"))
            per_group_records.append({
                "img": img, "condition": cond, "trial_number": trial_num,
                "NSS_intact_img": float("nan"), "NSS_scrambled_img": float("nan"),
                "NSS_diff_img": float("nan"), "n_subjects": n_subj,
            })

        if n_subj < int(min_subj_per_group):
            _append_nan()
            continue

        # Look up disambiguation reference maps
        if blend:
            fm_intact    = fm_index.get((img, cond, "disamb_intact"))
            fm_scrambled = fm_index.get((img, cond, "disamb_not_intact"))
        else:
            fm_intact    = fm_index.get((img, cond, "disamb_intact",     "ALL_TRIALS"))
            fm_scrambled = fm_index.get((img, cond, "disamb_not_intact", "ALL_TRIALS"))

        def _safe_ref(fm):
            if fm is None:
                return None
            ref = fm.get("fixMapPerIm")
            return ref if (ref is not None and np.ndim(ref) == 2) else None

        ref_maps = {"intact": _safe_ref(fm_intact), "scrambled": _safe_ref(fm_scrambled)}

        zrefs = {}
        for k, ref in ref_maps.items():
            if ref is None:
                zrefs[k] = None
                if DEBUG:
                    t_str = f" trial={trial_num}" if trial_num is not None else ""
                    print(f"  [cross] ❌ Missing {k} reference  img={img}{t_str}")
            else:
                zref, _, _ = _z_normalize(np.asarray(ref, dtype=float))
                zrefs[k] = zref

        subj_out              = []
        subj_scores_intact    = []
        subj_scores_scrambled = []

        for j in range(n_subj):
            coords_j = coords_list[j] if j < len(coords_list) \
                       else (np.array([], np.int32), np.array([], np.int32))
            nss_i = _nss_for_subject(zrefs["intact"],    coords_j, dy_off, dx_off)
            nss_s = _nss_for_subject(zrefs["scrambled"], coords_j, dy_off, dx_off)
            nss_d = nss_i - nss_s if np.isfinite(nss_i) and np.isfinite(nss_s) else float("nan")

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


# ── SUMMARY & PLOT ────────────────────────────────────────────────────────────

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
        out = pd.DataFrame(columns=["condition", "ref_type", "n_groups",
                                     "mean_NSS", "std_NSS", "ste_NSS"])
    else:
        kept = df[np.isfinite(df["NSS_img"])].copy()
        grp  = kept.groupby(["condition", "ref_type"], dropna=False)
        out  = grp.agg(
            n_groups=("img", "nunique"),
            mean_NSS=("NSS_img", "mean"),
            std_NSS =("NSS_img", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else np.nan),
        ).reset_index()
        out["ste_NSS"] = out["std_NSS"] / np.sqrt(out["n_groups"].clip(lower=1))

    out = out.sort_values(["condition", "ref_type"]).reset_index(drop=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"  Saved cross-phase NSS summary → {out_path}")
    return out


def plot_nss_crossphase_summary(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    data_conds     = ["C", "U"]
    display_labels = ["Conscious", "Unconscious"]
    palette_ref    = {"intact": PALETTE[3], "scrambled": PALETTE[2]}
    df = df.set_index(["condition", "ref_type"]).sort_index()
    x, width = range(len(data_conds)), 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.spines[["top", "right"]].set_visible(False)
    for i, ref in enumerate(["intact", "scrambled"]):
        offset = (i - 0.5) * width
        vals = [df.loc[(c, ref), "mean_NSS"] if (c, ref) in df.index else np.nan for c in data_conds]
        errs = [df.loc[(c, ref), "ste_NSS"]  if (c, ref) in df.index else np.nan for c in data_conds]
        ax.bar([p + offset for p in x], vals, width=width, yerr=errs, capsize=5,
               alpha=0.5, color=palette_ref[ref], label=ref.capitalize())
    ax.set_ylabel("Mean NSS")
    ax.set_title("Cross-phase NSS: Mooney vs Disambiguation")
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.legend(title="Reference")
    plt.tight_layout()
    out = FIGURES_DIR / Path(parquet_path).with_suffix(".png").name
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved plot → {out}")


# ── JAMOVI EXPORT ─────────────────────────────────────────────────────────────

def _flatten_cross_phase(CrossResults: dict) -> pd.DataFrame:
    rows = []
    for img_data in CrossResults["image"]:
        for subj in img_data["subject"]:
            if "ParticipantID" not in subj:
                continue
            rows.append({
                "Participant":   subj["ParticipantID"],
                "Image":         img_data["img"],
                "Session":       img_data["condition"],
                "trial_number":  img_data.get("trial_number"),
                "NSS_Intact":    subj["NSS_intact"],
                "NSS_Scrambled": subj["NSS_scrambled"],
            })
    return pd.DataFrame(rows)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_cross_phase_nss(blend: bool = BLEND_TRIALS):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    mode_label = "BLEND" if blend else "PER-TRIAL"
    print(f"\n{'='*60}")
    print(f"  Cross-Phase NSS  [{mode_label} mode]")
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
    cache_path = OUTPUT_DIR / f"NSS_CrossPhase_{'blend' if blend else 'pertrial'}.pkl"
    meta = _meta_block(ppd, IMAGE_HEIGHT, IMAGE_WIDTH, gcols,
                       tag="calculate_NSS_crossphase:v3",
                       extra={"nan_policy": NAN_POLICY_CROSS,
                              "min_subj_per_group": MIN_SUBJ_PER_GROUP_CROSS,
                              "ref_blending": "ALL_TRIALS" if not blend else "per_trial"})
    try:
        with open(cache_path, "rb") as f:
            cc = pickle.load(f)
        if isinstance(cc, dict) and cc.get("meta") == meta:
            CrossResults = cc["data"]
            print(f"  Loaded cross-phase NSS from cache → {cache_path}")
        else:
            raise ValueError("meta changed")
    except Exception:
        print("  Computing cross-phase NSS…")
        CrossResults = calculate_NSS_crossphase(
            FixMaps, fixations, ppd, blend=blend,
            nan_policy=NAN_POLICY_CROSS,
            min_subj_per_group=MIN_SUBJ_PER_GROUP_CROSS,
        )
        with open(cache_path, "wb") as f:
            pickle.dump({"meta": meta, "data": CrossResults}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved cross-phase NSS → {cache_path}")

    for label, key in [("Intact",    "meanNSS_intact_per_image"),
                        ("Scrambled", "meanNSS_scrambled_per_image"),
                        ("Diff",      "meanNSS_diff_per_image")]:
        arr = np.asarray(CrossResults[key], dtype=float)
        n   = np.isfinite(arr).sum()
        print(f"  Cross-phase {label}: {n} groups kept, mean={np.nanmean(arr):.4f}")

    # ── Summary parquet + plot ─────────────────────────────────────────────────
    summary_path = OUTPUT_DIR / "NSS_crossphase_summary.parquet"
    summarise_nss_crossphase_to_parquet(CrossResults, summary_path)
    plot_nss_crossphase_summary(summary_path)

    # ── Long-format CSV for Jamovi LMM ─────────────────────────────────────────
    df_c = _flatten_cross_phase(CrossResults)
    if blend:
        df_c = df_c.drop(columns=["trial_number"], errors="ignore")
    # --- ADD MOONEY AWARENESS ---
    aw = fixations[fixations["image_type"] == "mooney_post_intact"].rename(columns={"participant": "Participant", "ImageName": "Image", "session": "Session"})
    m_cols = ["Participant", "Image", "Session"] + (["trial_number"] if "trial_number" in df_c.columns else [])
    df_c = df_c.merge(aw[m_cols + ["awareness"]].drop_duplicates(), on=m_cols, how="left")

    id_vars = ["Participant", "Image", "Session", "awareness"] + \
              (["trial_number"] if "trial_number" in df_c.columns else [])
    df_c_melted = df_c.melt(
        id_vars=id_vars,
        value_vars=["NSS_Intact", "NSS_Scrambled"],
        var_name="ReferenceMap",
        value_name="NSS",
    )
    df_c_melted["ReferenceMap"] = df_c_melted["ReferenceMap"].str.replace("NSS_", "")
    df_c_melted.to_csv(OUTPUT_DIR / "NSS_CrossPhase_LongFormat.csv", index=False)
    print(f"  ✅ Cross-phase long format → {OUTPUT_DIR}/NSS_CrossPhase_LongFormat.csv")