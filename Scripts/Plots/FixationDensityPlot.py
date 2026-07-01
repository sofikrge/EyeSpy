"""
FixationDensityPlot.py
----------------------
Aggregate fixation density heatmap across all images, split by awareness state
and image type.

3x3 grid:
    Rows    - awareness state: Conscious Aware / Unconscious Aware / Unconscious Unaware
    Columns - image type:      Post-Intact Mooney / Intact Disambiguator / Scrambled Disambiguator

Each panel is built using the EXACT same pipeline as CreateFixationMaps_from_df in NSS.py:
  1. Per participant per image  → uint32 hit map (bincount of fixation pixels)
  2. Sum participant hit maps   → divide by n_subjects  →  per-image average map
  3. Average per-image maps across all images in the cell
  4. gaussian_filter(sigma=PPD/2, mode='reflect', truncate=2.0)
     (blur is linear so blurring the average == averaging the blurred maps - one blur at the end)

Diagnostic line at the bottom of each panel:
    Left %  - fraction of in-bounds fixations in the left  half of the canvas (w_pixel <= WIDTH/2)
    Top  %  - fraction of in-bounds fixations in the top   half of the canvas (h_pixel <= HEIGHT/2)
    n       - total in-bounds fixation events summed across all images in the cell
              (raw count, not subject-averaged - gives a sense of data volume)

Run from the project root:
    python3 Scripts/Analysis/FixationDensityPlot.py

Requires: data/NSS_all_fixations_clean.parquet  (produced by NSSExporter.py)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================

FIX_FILE    = Path("data/NSS_all_fixations_clean.parquet")
OUTPUT_DIR  = Path("Figures/nss_separated_analyses")
OUTPUT_FILE = OUTPUT_DIR / "FixationDensityPlot4.png"

IMAGE_HEIGHT = 600    # eye-tracking canvas height in pixels - must match NSS.py
IMAGE_WIDTH  = 800    # eye-tracking canvas width  in pixels - must match NSS.py
MASK_PPD     = 48.55  # pixels per visual degree - must match NSS.py
SIGMA        = MASK_PPD / 2.0  # Gaussian blur radius - identical to NSS.py

COLORMAP    = "jet"
DPI         = 180
FIGURE_SIZE = (15, 12)

# =============================================================================
# GRID DEFINITION
# =============================================================================

AWARENESS_ROWS = [
    ("conscious_aware",     "Conscious Aware\n(PAS 2-3)"),
    ("unconscious_aware",   "Unconscious Aware\n(PAS 2-3)"),
    ("unconscious_unaware", "Unconscious Unaware\n(PAS 0)"),
]

IMAGE_TYPE_COLS = [
    ("mooney_post_intact", "Post-Intact Mooney"),
    ("disamb_intact",      "Intact Disambiguator"),
    ("disamb_not_intact",  "Scrambled Disambiguator"),
]

# Disambiguator maps are session-split (not awareness-split) - same as NSS.py reference maps.
# Conscious session = C, both unconscious groups share session U.
AWARENESS_TO_SESSION = {
    "conscious_aware":     "C",
    "unconscious_aware":   "U",
    "unconscious_unaware": "U",
}
SESSION_LABEL = {"C": "Session C  (Conscious)", "U": "Session U  (Unconscious)"}

# =============================================================================
# COORDINATE HELPERS - exact mirror of NSS.py
# =============================================================================

def _round_half_away_from_zero(x):
    """Ties round away from zero - matches MATLAB round() and NSS.py."""
    x = np.asarray(x, dtype=float)
    return (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(int)


def _deg_to_image_pixels(x_deg, y_deg):
    """Convert visual degrees to 1-based pixel coordinates.

    Returns (h_1based, w_1based) = (row, col) - same signature as NSS.py's
    _deg_to_image_pixels so index arithmetic below is identical.
    """
    w_1b = _round_half_away_from_zero((IMAGE_WIDTH  / 2.0) + np.asarray(x_deg, float) * MASK_PPD)
    h_1b = _round_half_away_from_zero((IMAGE_HEIGHT / 2.0) + np.asarray(y_deg, float) * MASK_PPD)
    return h_1b, w_1b  # (row, col) 1-based


# =============================================================================
# MAP BUILDING - replicates CreateFixationMaps_from_df, aggregated across images
# =============================================================================

def build_aggregate_density_map(df_cell):
    """Build an aggregate fixation density map for one (awareness × image_type) cell.

    Follows CreateFixationMaps_from_df step by step:
      • Per participant: binary hit map (uint32, same as NSS.py)
      • Sum participants → float32, divide by n_subjects  →  per-image average map
      • Accumulate per-image averages, divide by n_images  →  cross-image aggregate
      • One gaussian_filter call at the end (equivalent to averaging blurred per-image
        maps because the filter is a linear operation)

    Returns
    -------
    density : float32 array [IMAGE_HEIGHT × IMAGE_WIDTH]
        Blurred aggregate map, unscaled.
    n_in_bounds : int
        Total in-bounds fixation events across all participants and images in this cell
        (raw sum, not subject-averaged - used only for the diagnostic label).
    """
    H, W = IMAGE_HEIGHT, IMAGE_WIDTH
    aggregate   = np.zeros((H, W), dtype=np.float64)
    n_images    = 0
    n_in_bounds = 0

    for (_img, _session), df_img in df_cell.groupby(["ImageName", "session"], dropna=False):
        image_sum = np.zeros((H, W), dtype=np.float64)
        n_subj    = 0

        for _pid, df_subj in df_img.groupby("participant", dropna=False):
            xdeg = pd.to_numeric(df_subj["x_deg_centered"], errors="coerce").to_numpy()
            ydeg = pd.to_numeric(df_subj["y_deg"],          errors="coerce").to_numpy()
            ok   = np.isfinite(xdeg) & np.isfinite(ydeg)

            hit_map = np.zeros((H, W), dtype=np.uint32)
            if ok.any():
                h1, w1 = _deg_to_image_pixels(xdeg[ok], ydeg[ok])
                inside = (h1 >= 1) & (h1 <= H) & (w1 >= 1) & (w1 <= W)
                if inside.any():
                    h0  = h1[inside] - 1          # 0-based row
                    w0  = w1[inside] - 1          # 0-based col
                    lin = (h0 * W + w0).astype(np.int64)
                    # cast to uint16 before adding - mirrors NSS.py exactly
                    hit_map += np.bincount(lin, minlength=H * W).reshape(H, W).astype(np.uint16)
                    n_in_bounds += int(inside.sum())

            image_sum += hit_map.astype(np.float64)
            n_subj    += 1

        if n_subj > 0:
            # Divide by n_subjects - matches: mapPerIm /= float(nsubj)
            aggregate += image_sum / float(n_subj)
            n_images  += 1

    if n_images == 0:
        return np.zeros((H, W), dtype=np.float32), 0

    # Divide by n_images - turns sum-of-per-image-averages into a grand average
    aggregate /= float(n_images)

    # Single blur - equivalent to averaging already-blurred per-image maps
    # Parameters: same as gaussian_filter(mapPerIm, sigma=sigma, mode="reflect", truncate=2.0)
    density = gaussian_filter(aggregate.astype(np.float32), sigma=SIGMA,
                              mode="reflect", truncate=2.0)
    return density, n_in_bounds


def _compute_split_percentages(df_cell):
    """Fraction of in-bounds fixations in the left and top canvas halves.

    Computed from the same pixel coordinates as the density map so that the
    numbers correspond exactly to the visual split shown by the crosshair.
      Left half : w_1based <= IMAGE_WIDTH  / 2
      Top  half : h_1based <= IMAGE_HEIGHT / 2
    """
    H, W = IMAGE_HEIGHT, IMAGE_WIDTH
    xdeg = pd.to_numeric(df_cell["x_deg_centered"], errors="coerce").to_numpy()
    ydeg = pd.to_numeric(df_cell["y_deg"],          errors="coerce").to_numpy()
    ok   = np.isfinite(xdeg) & np.isfinite(ydeg)
    if not ok.any():
        return np.nan, np.nan

    h1, w1 = _deg_to_image_pixels(xdeg[ok], ydeg[ok])
    inside = (h1 >= 1) & (h1 <= H) & (w1 >= 1) & (w1 <= W)
    if not inside.any():
        return np.nan, np.nan

    n        = inside.sum()
    pct_left = 100.0 * (w1[inside] <= W / 2).sum() / n
    pct_top  = 100.0 * (h1[inside] <= H / 2).sum() / n
    return pct_left, pct_top


# =============================================================================
# PLOTTING
# =============================================================================

def _style_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, IMAGE_WIDTH)
    ax.set_ylim(IMAGE_HEIGHT, 0)   # origin upper-left, y increases downward
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_panel(ax, density, n_in_bounds, pct_left, pct_top):
    """Render density map and diagnostic annotation into one panel."""
    norm = density / density.max() if density.max() > 0 else density
    ax.imshow(norm, cmap=COLORMAP, origin="upper",
              extent=[0, IMAGE_WIDTH, IMAGE_HEIGHT, 0],
              vmin=0, vmax=1, aspect="auto")

    # Faint crosshair at canvas centre - visual reference for the left/top split
    ax.axvline(IMAGE_WIDTH  / 2, color="white", linewidth=0.8, alpha=0.5, linestyle="--")
    ax.axhline(IMAGE_HEIGHT / 2, color="white", linewidth=0.8, alpha=0.5, linestyle="--")

    if np.isnan(pct_left):
        label = "No data"
    else:
        label = f"Left {pct_left:.1f}%  ·  Top {pct_top:.1f}%  ·  n = {n_in_bounds:,}"
    ax.text(0.5, 0.03, label,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, color="#111111",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.65, linewidth=0))


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading fixations…")
    fix = pd.read_parquet(FIX_FILE)
    fix["x_deg_centered"] = pd.to_numeric(fix["x_deg_centered"], errors="coerce")
    fix["y_deg"]          = pd.to_numeric(fix["y_deg"],          errors="coerce")
    fix = fix.dropna(subset=["x_deg_centered", "y_deg"])
    print(f"  {len(fix):,} fixations loaded after dropping NaN coordinates.\n")

    fig, axes = plt.subplots(3, 3, figsize=FIGURE_SIZE)
    fig.patch.set_facecolor("white")

    for row_idx, (awareness_key, awareness_label) in enumerate(AWARENESS_ROWS):
        session_key = AWARENESS_TO_SESSION[awareness_key]

        for col_idx, (img_type_key, img_type_label) in enumerate(IMAGE_TYPE_COLS):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("white")

            is_mooney = img_type_key == "mooney_post_intact"

            if is_mooney:
                # Mooney: filter by awareness group (awareness-split, as in NSS.py)
                subset = fix[
                    (fix["awareness"]  == awareness_key) &
                    (fix["image_type"] == img_type_key)
                ]
            else:
                # Disambiguator: filter by session only - matches NSS.py reference map construction
                # (disamb maps are NOT awareness-split; both UA and UU rows use session U)
                subset = fix[
                    (fix["session"]    == session_key) &
                    (fix["image_type"] == img_type_key)
                ]

            if subset.empty:
                ax.text(0.5, 0.5, "No data", color="#333333", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11)
                _style_axes(ax)
            else:
                filter_desc = awareness_key if is_mooney else SESSION_LABEL[session_key]
                print(f"  Building map: [{filter_desc:35s} × {img_type_key}]")
                density, n_in_bounds = build_aggregate_density_map(subset)
                pct_left, pct_top    = _compute_split_percentages(subset)

                _draw_panel(ax, density, n_in_bounds, pct_left, pct_top)
                _style_axes(ax)

                # Stamp the session on disamb panels so it's clear which session is shown
                # (both UA and UU rows show the same session U disamb maps)
                if not is_mooney:
                    ax.text(0.98, 0.97, SESSION_LABEL[session_key],
                            transform=ax.transAxes, ha="right", va="top",
                            fontsize=7.5, color="#555555",
                            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                      alpha=0.75, linewidth=0))

                print(f"    n_in_bounds={n_in_bounds:>7,}  left={pct_left:5.1f}%  top={pct_top:5.1f}%")

            if row_idx == 0:
                ax.set_title(img_type_label, fontsize=13, fontweight="bold",
                             color="#111111", pad=10)
            if col_idx == 0:
                ax.set_ylabel(awareness_label, fontsize=11, fontweight="bold",
                              color="#111111", labelpad=14)

    fig.suptitle("Fixation Density Aggregated Across All Images",
                 fontsize=16, color="#111111", y=0.995)
    fig.subplots_adjust(hspace=0.07, wspace=0.05)

    fig.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
