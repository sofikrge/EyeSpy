"""
Mooney-on-Disambiguator Visualization
--------------------------------------
A more *direct* way of looking at the cross-phase NSS than NSSSVisualiser.py.

The cross-phase NSS score is literally: "how well do the Mooney-phase fixations land on
the disambiguation-phase saliency map?" — scored once against the INTACT reference map and
once against the SCRAMBLED reference map (NSS_diff = intact - scrambled).

Instead of showing three independent heatmaps and asking the viewer to mentally compute the
overlap, this script overlays the actual Mooney fixation LOCATIONS on top of the intact and
scrambled reference maps side by side. If the Mooney fixations sit on the hot regions of the
scrambled map more than the intact map, you can SEE why NSS_scrambled > NSS_intact.

Per image (3x1 grid), focused on the unconscious_unaware condition:
    [ Mooney + its fixations ] [ Intact ref map + Mooney fixations ] [ Scrambled ref map + Mooney fixations ]

Run AFTER NSS.py has produced the cached pickles.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from scipy.fft import fft2, ifft2

# ============================================================================
# CONFIGURATION - Settings you might need to change
# ============================================================================

# 1. Where to find the data and save the plots
BASE_PATH = Path("analysesresults/NSS")
STATS_PATH = BASE_PATH / "NSS_crossphase_descriptives.pkl"  # Cross-phase scores (ranking + DV)
MAPS_PATH = BASE_PATH / "FixMaps_full.pkl"                  # Pre-blurred reference saliency maps
FIX_FILE = Path("data/NSS_all_fixations_clean.parquet")     # Raw fixations (for the overlay points)
OUTPUT_DIR = Path("Figures/nss_separated_analyses/MooneysOnDisamb")  # Where images will be saved

# 2. Screen and Eye-Tracking properties (kept consistent with NSS.py / Settings.py by hand)
MASK_PPD = 48.55          # Pixels Per Degree
IMAGE_HEIGHT = 600        # Eye-tracking capture height
IMAGE_WIDTH = 800         # Eye-tracking capture width
IMAGE_SIZE_DEG = (9.99, 7.50)  # On-screen stimulus size (width, height) in degrees

# 3. Where the original background images are stored on your computer
MOONEY_DIRS = [
    Path('/Users/sofiakarageorgiou/Documents/GitHub/Experiment Code/RUN_ME/Stimuli/ImageTrials_Experiment'),
    Path('/Users/sofiakarageorgiou/Documents/GitHub/Experiment Code/RUN_ME/Stimuli/ImageTrials_ExtraTrials')
]
DISAMB_DIRS = [
    Path('/Users/sofiakarageorgiou/Documents/GitHub/Experiment Code/RUN_ME/Stimuli/ImageDisamb_Experiment'),
    Path('/Users/sofiakarageorgiou/Documents/GitHub/Experiment Code/RUN_ME/Stimuli/ImageDisamb_ExtraTrials')
]

# 4. Visual settings
N_TOP = 40                # How many top-ranked images to plot
FIGURE_SIZE = (20, 18)    # Width, Height of the final saved image
DPI = 150
HEATMAP_COLORMAP = "jet"  # Reference saliency map colors (blue=low, red=high)
HEATMAP_ALPHA = 0.45      # Transparency of the reference map overlay
HEATMAP_THRESHOLD = 0.05  # Hide reference-map values below this (fraction of max) to cut border noise
FIX_POINT_COLOR = "lime"  # Mooney fixation marker color
FIX_POINT_SIZE = 22       # Mooney fixation marker size
FIX_POINT_ALPHA = 0.85    # Mooney fixation marker transparency

# ============================================================================
# COORDINATE HELPERS - replicated from NSS.py so this script is standalone
# ============================================================================

def round_half_away_from_zero(x):
    """Round to nearest integer, ties away from zero (matches MATLAB's round / NSS.py)."""
    x = np.asarray(x, dtype=float)
    return (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(int)

def deg_to_image_pixels(x_deg, y_deg, ppd, *, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    """Convert visual degrees to image pixel coords (col, row), matching NSS.py mapping.

    NSS.py builds maps as map[h=row, w=col] with degree-0 at the screen centre, so we
    reproduce the exact same transform here for the scatter overlay to line up 1:1."""
    w = round_half_away_from_zero((width / 2.0) + np.asarray(x_deg, float) * ppd)
    h = round_half_away_from_zero((height / 2.0) + np.asarray(y_deg, float) * ppd)
    return w, h  # x (col), y (row)

# ============================================================================
# IMAGE / DATA HELPERS - mirror NSSSVisualiser.py behaviour
# ============================================================================

def find_image_file(img_name, directories):
    """Looks through a list of folders to find the requested image file."""
    filename = img_name if str(img_name).lower().endswith('.jpg') else f"{img_name}.jpg"
    for directory in directories:
        filepath = directory / filename
        if filepath.exists():
            return filepath
    return None

def scramble_image(image_path):
    """Create a representative Fourier-scrambled version of an image.

    NOTE: this is a *representation* of a scramble, not necessarily the exact stimulus the
    participant saw. The scrambled REFERENCE MAP overlaid on top is real gaze data; the
    background here is only for context."""
    img = mpimg.imread(image_path)
    if img.dtype == np.uint8:
        img = img.astype(float) / 255.0
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    h, w, c = img.shape
    np.random.seed(42)
    random_phase = np.angle(fft2(np.random.rand(h, w)))
    scrambled = np.zeros_like(img)
    for channel in range(c):
        fft_result = fft2(img[:, :, channel])
        amplitude = np.abs(fft_result)
        phase = np.angle(fft_result)
        new_fft = amplitude * np.exp(1j * (phase + random_phase))
        scrambled[:, :, channel] = np.real(ifft2(new_fft))
    scrambled = (scrambled - scrambled.min()) / (scrambled.max() - scrambled.min())
    scrambled = img.min() + (img.max() - img.min()) * scrambled
    return np.clip(scrambled, 0, 1)

def get_fixation_map(fixation_maps, image_name, session, image_type):
    """Searches the loaded FixMaps for the specific pre-calculated reference map."""
    for fmap in fixation_maps:
        if str(fmap['img']) == str(image_name) and \
           str(fmap['condition']).upper() == str(session).upper() and \
           fmap['image_type'] == image_type:
            return fmap['fixMapPerIm']
    return None

def load_data():
    """Loads cross-phase stats, reference maps, and the raw fixations."""
    print("📂 Loading data...")
    with open(STATS_PATH, "rb") as f:
        stats = pickle.load(f)
    results = stats["data"] if isinstance(stats, dict) else stats

    with open(MAPS_PATH, "rb") as f:
        maps = pickle.load(f)["data"]

    fixations = pd.read_parquet(FIX_FILE)
    return results, maps, fixations

def rank_images_by_interaction(results):
    """Ranks images by NSS_diff (NSS_intact - NSS_scrambled) within the unconscious_unaware
    condition, ascending — so the MOST NEGATIVE diffs come first. A negative diff means the
    scrambled reference map scores higher than the intact one (scrambled > intact), which is
    the effect of interest here. Returns (diff, intact, scrambled, img) tuples."""
    ranked = []
    for entry in results['image']:
        if entry.get('condition') != 'U':
            continue
        if entry.get('awareness') != 'unconscious_unaware':
            continue
        diff = entry.get('NSS_diff_img', np.nan)
        if np.isfinite(diff):
            ranked.append((diff,
                           entry.get('NSS_intact_img', np.nan),
                           entry.get('NSS_scrambled_img', np.nan),
                           entry['img']))
    ranked.sort(key=lambda t: t[0], reverse=False)
    return ranked

def build_score_lookup(results):
    """(img, awareness) -> (condition, NSS_intact, NSS_scrambled, NSS_diff) from cross-phase results."""
    lookup = {}
    for entry in results['image']:
        lookup[(entry['img'], entry.get('awareness'))] = (
            entry.get('condition'),
            entry.get('NSS_intact_img', np.nan),
            entry.get('NSS_scrambled_img', np.nan),
            entry.get('NSS_diff_img', np.nan),
        )
    return lookup

def get_mooney_fixation_points(fixations, image_name, session, awareness):
    """Pixel coords (x, y) of the Mooney fixations for one image / session / awareness.

    These are the SAME fixations the cross-phase NSS scores against the disamb maps."""
    mask = (
        (fixations["ImageName"].astype(str) == str(image_name)) &
        (fixations["session"].astype(str).str.upper() == str(session).upper()) &
        (fixations["image_type"] == "mooney_post_intact") &
        (fixations["awareness"] == awareness)
    )
    sub = fixations[mask]
    if sub.empty:
        return np.array([]), np.array([])
    xdeg = pd.to_numeric(sub["x_deg_centered"], errors="coerce").to_numpy()
    ydeg = pd.to_numeric(sub["y_deg"], errors="coerce").to_numpy()
    ok = np.isfinite(xdeg) & np.isfinite(ydeg)
    x_px, y_px = deg_to_image_pixels(xdeg[ok], ydeg[ok], MASK_PPD)
    inside = (x_px >= 1) & (x_px <= IMAGE_WIDTH) & (y_px >= 1) & (y_px <= IMAGE_HEIGHT)
    return x_px[inside], y_px[inside]

# ============================================================================
# PLOTTING
# ============================================================================

def _draw_background(ax, background, img_type, img_extent):
    """Draws a stimulus image (or pre-scrambled array) centred in the canvas."""
    if background is None:
        return
    if isinstance(background, (str, Path)):
        img = mpimg.imread(background)
        cmap = 'gray' if 'mooney' in img_type else None
        ax.imshow(img, origin='upper', cmap=cmap, extent=img_extent)
    else:
        ax.imshow(background, origin='upper', extent=img_extent)

def _draw_reference_map(ax, heatmap):
    """Overlays a reference saliency map, normalized to its own peak, low values masked out."""
    if heatmap is None:
        return
    norm = heatmap / heatmap.max() if heatmap.max() > 0 else heatmap
    masked = np.ma.masked_where(norm < HEATMAP_THRESHOLD, norm)
    ax.imshow(masked, cmap=HEATMAP_COLORMAP, alpha=HEATMAP_ALPHA,
              origin='upper', extent=[0, IMAGE_WIDTH, IMAGE_HEIGHT, 0])

def _draw_fixations(ax, x_px, y_px):
    """Scatters the Mooney fixation points on top of whatever is already drawn."""
    if x_px.size == 0:
        return
    ax.scatter(x_px, y_px, s=FIX_POINT_SIZE, c=FIX_POINT_COLOR,
               edgecolors='black', linewidths=0.4, alpha=FIX_POINT_ALPHA, zorder=5)

def _panel_score(ax, value):
    """Annotates a panel with its NSS score in the top-left corner."""
    ax.text(0.02, 0.98, f"NSS = {value:.3f}", transform=ax.transAxes,
            va='top', ha='left', fontsize=12, color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.55, pad=0.3), zorder=6)

def create_visualization(image_name, diff_uu, fixation_maps, fixations, score_lookup):
    """Builds the 3x3 figure for a single image.

    Rows = awareness conditions, Columns = Mooney / Intact ref map / Scrambled ref map.
    The SAME row's Mooney fixations are overlaid on that row's intact and scrambled maps."""
    # 1. Backgrounds (shared across rows)
    mooney_bg = find_image_file(image_name, MOONEY_DIRS)
    intact_bg = find_image_file(image_name, DISAMB_DIRS)
    scrambled_bg = scramble_image(intact_bg) if intact_bg else None

    # 2. Centre the stimulus image inside the 800x600 canvas (matches Settings.IMAGE_SIZE_DEG)
    img_px_w = IMAGE_SIZE_DEG[0] * MASK_PPD
    img_px_h = IMAGE_SIZE_DEG[1] * MASK_PPD
    x_pad = (IMAGE_WIDTH - img_px_w) / 2
    y_pad = (IMAGE_HEIGHT - img_px_h) / 2
    img_extent = [x_pad, IMAGE_WIDTH - x_pad, IMAGE_HEIGHT - y_pad, y_pad]

    # Rows: (row label, session, awareness). Disamb maps are NOT awareness-split, so they
    # depend only on the session (C vs U).
    rows = [
        ("Conscious Aware", "C", "conscious_aware"),
        ("UC Aware", "U", "unconscious_aware"),
        ("UC Unaware", "U", "unconscious_unaware"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=FIGURE_SIZE)

    for row_idx, (row_label, session, awareness) in enumerate(rows):
        # Reference maps for this session + this row's Mooney map
        intact_map = get_fixation_map(fixation_maps, image_name, session, "disamb_intact")
        scrambled_map = get_fixation_map(fixation_maps, image_name, session, "disamb_not_intact")
        mooney_map = get_fixation_map(fixation_maps, image_name, session, f"mooney_post_intact_{awareness}")

        # This row's Mooney fixation points (the thing being scored)
        x_px, y_px = get_mooney_fixation_points(fixations, image_name, session, awareness)

        # This row's NSS scores
        _, nss_intact, nss_scrambled, _ = score_lookup.get(
            (image_name, awareness), (None, np.nan, np.nan, np.nan))

        # --- Col 0: Mooney image + Mooney fixations (context) ---
        ax = axes[row_idx, 0]
        _draw_background(ax, mooney_bg, "mooney", img_extent)
        _draw_reference_map(ax, mooney_map)
        _draw_fixations(ax, x_px, y_px)

        # --- Col 1: Intact reference map + Mooney fixations ---
        ax = axes[row_idx, 1]
        _draw_background(ax, intact_bg, "disamb_intact", img_extent)
        _draw_reference_map(ax, intact_map)
        _draw_fixations(ax, x_px, y_px)
        _panel_score(ax, nss_intact)

        # --- Col 2: Scrambled reference map + Mooney fixations ---
        ax = axes[row_idx, 2]
        _draw_background(ax, scrambled_bg, "disamb_not_intact", img_extent)
        _draw_reference_map(ax, scrambled_map)
        _draw_fixations(ax, x_px, y_px)
        _panel_score(ax, nss_scrambled)

    # Lock every panel to the full canvas so the overlay lines up 1:1 with the maps
    column_titles = ["Mooney (where they looked)", "Intact ref map", "Scrambled ref map"]
    for row_idx, (row_label, _, _) in enumerate(rows):
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            ax.set_xlim(0, IMAGE_WIDTH)
            ax.set_ylim(IMAGE_HEIGHT, 0)  # origin at top, y increases downward
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_idx == 0:
                ax.set_title(column_titles[col_idx], fontsize=15, fontweight='bold', pad=12)
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=15, fontweight='bold', labelpad=20)

    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.suptitle(
        f"Image: {image_name}   |   ranked by unconscious_unaware NSS_diff (intact - scrambled) = {diff_uu:.3f}",
        fontsize=18, y=0.93
    )
    return fig

# ============================================================================
# MAIN
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results, fixation_maps, fixations = load_data()
    score_lookup = build_score_lookup(results)
    ranked_images = rank_images_by_interaction(results)
    top = ranked_images[:N_TOP]

    print(f"\n🎨 Generating {len(top)} visualizations...")

    for rank, (diff, nss_intact, nss_scrambled, img_name) in enumerate(top, start=1):
        print(f"   [{rank:2d}/{len(top)}] Processing Image: {img_name}")
        fig = create_visualization(img_name, diff, fixation_maps, fixations, score_lookup)
        output_file = OUTPUT_DIR / f"Rank_{rank:02d}_{img_name}.png"
        fig.savefig(output_file, dpi=DPI, bbox_inches='tight', transparent=False)
        plt.close(fig)

    print(f"\n✅ Done! All images saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
