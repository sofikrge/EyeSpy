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

Per image (3x6 grid): the standard 3x3 grid (rows = awareness, cols = Mooney / Intact /
Scrambled) is plotted twice side by side, once for left-eye-dominant participants and once
for right-eye-dominant participants. EVERYTHING is eye-split — the reference saliency maps,
the Left/Top bias % diagnostics, the Mooney fixation overlay, and the cross-phase NSS score.
Each half is therefore a fully self-contained cross-phase analysis for that eye group.

The eye-split reference maps and scores are recomputed on the fly from the parquet using
NSS.py's own functions (CreateFixationMaps_from_df + the cross-phase scoring helpers), so
the methodology matches the main analysis exactly. The cached cross-phase pickle is used
only to rank which images to plot (by the pooled unconscious_unaware NSS_diff).

Run AFTER NSS.py has produced the cached pickle (for ranking), and after Stage 1 +
NSSExporter have written a parquet that includes the `dominant_eye` column.
"""

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from scipy.fft import fft2, ifft2

# Reuse NSS.py's canonical map-building and scoring helpers so the eye-split maps and
# scores are computed with exactly the same methodology as the main analysis (LOSO-free
# cross-phase scoring, z-norm with ddof=0, disk stencil, MATLAB-compatible rounding).
_NSS_DIR = Path(__file__).resolve().parents[1] / "NSS"
if str(_NSS_DIR) not in sys.path:
    sys.path.insert(0, str(_NSS_DIR))
import NSS as nss  # noqa: E402

# ============================================================================
# CONFIGURATION - Settings you might need to change
# ============================================================================

# 1. Where to find the data and save the plots
BASE_PATH = Path("analysesresults/NSS")
STATS_PATH = BASE_PATH / "NSS_crossphase_descriptives.pkl"  # Cross-phase scores — used only to rank images
FIX_FILE = Path("data/NSS_all_fixations_clean.parquet")     # Raw fixations (overlay + eye-split maps/scores)
OUTPUT_DIR = Path("Figures/nss_separated_analyses/MooneysOnDisamb")  # Where images will be saved

# 2. Screen and Eye-Tracking properties (kept consistent with NSS.py / Settings.py by hand)
MASK_PPD = 48.55          # Pixels Per Degree
IMAGE_HEIGHT = 600        # Eye-tracking capture height
IMAGE_WIDTH = 800         # Eye-tracking capture width
IMAGE_SIZE_DEG = (9.99, 7.50)  # On-screen stimulus size (width, height) in degrees

# 3. Where the original background images are stored on your computer
MOONEY_DIRS = [
    Path('/Users/sofiakarageorgiou/Documents/GitHub/Experiment_Code/RUN_ME/Stimuli/ImageTrials_Experiment'),
    Path('/Users/sofiakarageorgiou/Documents/GitHub/Experiment_Code/RUN_ME/Stimuli/ImageTrials_ExtraTrials')
]
DISAMB_DIRS = [
    Path('/Users/sofiakarageorgiou/Documents/GitHub/Experiment_Code/RUN_ME/Stimuli/ImageDisamb_Experiment'),
    Path('/Users/sofiakarageorgiou/Documents/GitHub/Experiment_Code/RUN_ME/Stimuli/ImageDisamb_ExtraTrials')
]

# 4. Visual settings
N_TOP = 40                # How many top-ranked images to plot
FIGURE_SIZE = (46, 18)    # Width, Height of the final saved image (3x6: two eye groups side by side)
# Dominant-eye groups plotted side by side (left triplet, right triplet)
EYE_GROUPS = [("left", "Left-eye dominant"), ("right", "Right-eye dominant")]
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

def _fixation_subset(fixations, image_name, session, image_type, awareness=None, eye=None):
    """Filter the parquet to one image / session / image_type, optionally awareness + eye."""
    mask = (
        (fixations["ImageName"].astype(str) == str(image_name)) &
        (fixations["session"].astype(str).str.upper() == str(session).upper()) &
        (fixations["image_type"] == image_type)
    )
    if awareness is not None:
        mask &= (fixations["awareness"] == awareness)
    if eye is not None:
        mask &= (fixations["dominant_eye"].astype(str) == str(eye))
    return fixations[mask]

def build_reference_map(fixations, image_name, session, image_type, awareness=None, eye=None):
    """Build a single-image (eye-split) saliency map straight from the parquet using
    NSS.py's CreateFixationMaps_from_df, so it is identical to the analysis reference maps
    (per-subject hit map → average across subjects → Gaussian blur). Returns the blurred
    map array, or None if there are no matching fixations."""
    sub = _fixation_subset(fixations, image_name, session, image_type, awareness, eye)
    if sub.empty:
        return None
    fmaps = nss.CreateFixationMaps_from_df(sub, MASK_PPD)  # one group in, one entry out
    return fmaps[0]["fixMapPerIm"] if fmaps else None

def score_eye_cross(fixations, image_name, session, awareness, eye, intact_map, scrambled_map):
    """Cross-phase NSS for one eye group: that group's Mooney fixations scored against the
    (eye-split) intact / scrambled reference maps. Mirrors calculate_NSS_crossphase exactly
    — z-normalise each reference, score every (participant, trial) unit via the disk stencil,
    then aggregate with the permissive NaN policy. Returns (nss_intact, nss_scrambled)."""
    df_group = _fixation_subset(fixations, image_name, session, "mooney_post_intact",
                                awareness=awareness, eye=eye)
    if df_group.empty:
        return np.nan, np.nan

    dy_off, dx_off = nss._disk_offsets(nss.SIGMA)
    coords_list = nss._coords_in_fixmaps_order(df_group, MASK_PPD, IMAGE_HEIGHT, IMAGE_WIDTH)

    def _zref(m):
        if m is None:
            return None
        z, _, _ = nss._z_normalize(np.asarray(m, dtype=float))
        return z

    z_intact, z_scram = _zref(intact_map), _zref(scrambled_map)
    scores_i = [nss._nss_for_subject(z_intact, c, dy_off, dx_off) for c in coords_list]
    scores_s = [nss._nss_for_subject(z_scram, c, dy_off, dx_off) for c in coords_list]
    return (nss._aggregate_by_policy(scores_i, "permissive"),
            nss._aggregate_by_policy(scores_s, "permissive"))

def load_data():
    """Loads cross-phase stats (for ranking only) and the raw fixations."""
    print("📂 Loading data...")
    with open(STATS_PATH, "rb") as f:
        stats = pickle.load(f)
    results = stats["data"] if isinstance(stats, dict) else stats

    fixations = pd.read_parquet(FIX_FILE)
    return results, fixations

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

def get_fixation_points(fixations, image_name, session, image_type, awareness=None, eye=None):
    """Pixel coords (x, y) of fixations for one image / session / image_type.

    Mirrors how each panel's map is built:
      • Mooney maps are awareness-split  → pass awareness to filter the group.
      • Disambiguator maps are session-split, NOT awareness-split → leave awareness=None
        (matches NSS.py reference maps and FixationDensityPlot).
      • Pass eye ('left'/'right') to keep only participants with that dominant eye."""
    mask = (
        (fixations["ImageName"].astype(str) == str(image_name)) &
        (fixations["session"].astype(str).str.upper() == str(session).upper()) &
        (fixations["image_type"] == image_type)
    )
    if awareness is not None:
        mask &= (fixations["awareness"] == awareness)
    if eye is not None:
        mask &= (fixations["dominant_eye"].astype(str) == str(eye))
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
    """Labels a disambiguator panel with its cross-phase NSS score, centred just above it."""
    ax.text(0.5, 0.9, f"Cross-Phase NSS = {value:.3f}", transform=ax.transAxes,
            va='bottom', ha='center', fontsize=12, color='#111111', zorder=6)

def _panel_diagnostic(ax, x_px, y_px):
    """Bottom-centre label: Left%, Top%, n — same format as FixationDensityPlot."""
    if x_px.size == 0:
        label = "n = 0"
    else:
        n = x_px.size
        pct_left = 100.0 * (x_px <= IMAGE_WIDTH  / 2).sum() / n
        pct_top  = 100.0 * (y_px <= IMAGE_HEIGHT / 2).sum() / n
        label = f"Left {pct_left:.1f}%  ·  Top {pct_top:.1f}%  ·  n = {n:,}"
    ax.text(0.5, 0.09, label, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=12, color="#111111",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.65, linewidth=0),
            zorder=7)

def create_visualization(image_name, diff_uu, fixations):
    """Builds the 3x6 figure for a single image: the same 3x3 grid plotted once per
    dominant-eye group, side by side.

    Rows = awareness conditions. Within each eye group the columns are
    Mooney / Intact ref map / Scrambled ref map. EVERYTHING is eye-split — the reference
    maps, the Mooney overlay, the disambiguator-phase bias diagnostics, and the NSS scores
    are all computed from this eye group's fixations only."""
    # 1. Backgrounds (shared across rows AND eye groups)
    mooney_bg = find_image_file(image_name, MOONEY_DIRS)
    intact_bg = find_image_file(image_name, DISAMB_DIRS)
    scrambled_bg = scramble_image(intact_bg) if intact_bg else None

    # 2. Centre the stimulus image inside the 800x600 canvas (matches Settings.IMAGE_SIZE_DEG)
    img_px_w = IMAGE_SIZE_DEG[0] * MASK_PPD
    img_px_h = IMAGE_SIZE_DEG[1] * MASK_PPD
    x_pad = (IMAGE_WIDTH - img_px_w) / 2
    y_pad = (IMAGE_HEIGHT - img_px_h) / 2
    img_extent = [x_pad, IMAGE_WIDTH - x_pad, IMAGE_HEIGHT - y_pad, y_pad]

    # Rows: (row label, session, awareness). Disamb maps are session-split (not awareness-
    # split), but here they ARE further eye-split, so they depend on (session, eye).
    rows = [
        ("Conscious Aware", "C", "conscious_aware"),
        ("UC Aware", "U", "unconscious_aware"),
        ("UC Unaware", "U", "unconscious_unaware"),
    ]

    # Cache eye-split maps within this image so the shared session-U disamb maps
    # (UA + UU rows) are only built once per eye.
    _map_cache = {}
    def ref_map(image_type, session, eye, awareness=None):
        key = (image_type, str(session).upper(), eye, awareness)
        if key not in _map_cache:
            _map_cache[key] = build_reference_map(
                fixations, image_name, session, image_type, awareness=awareness, eye=eye)
        return _map_cache[key]

    fig, axes = plt.subplots(3, 6, figsize=FIGURE_SIZE)

    for eye_idx, (eye, _eye_title) in enumerate(EYE_GROUPS):
        col_off = eye_idx * 3  # 0 for the left triplet, 3 for the right triplet

        for row_idx, (row_label, session, awareness) in enumerate(rows):
            # Eye-split reference maps for this session + this row's eye-split Mooney map
            intact_map    = ref_map("disamb_intact", session, eye)
            scrambled_map = ref_map("disamb_not_intact", session, eye)
            mooney_map    = ref_map("mooney_post_intact", session, eye, awareness=awareness)

            # This eye group's Mooney fixation points (the thing being scored, overlaid on every column)
            x_px, y_px = get_fixation_points(
                fixations, image_name, session, "mooney_post_intact", awareness, eye=eye)

            # This eye group's disambiguator-phase fixations that BUILT each reference map
            # (session + eye split) — used only for that panel's bias diagnostic.
            intact_x, intact_y       = get_fixation_points(fixations, image_name, session, "disamb_intact", eye=eye)
            scrambled_x, scrambled_y = get_fixation_points(fixations, image_name, session, "disamb_not_intact", eye=eye)

            # This eye group's NSS scores, recomputed against this eye group's reference maps
            nss_intact, nss_scrambled = score_eye_cross(
                fixations, image_name, session, awareness, eye, intact_map, scrambled_map)

            # --- Col 0: Mooney image + Mooney fixations (context) ---
            ax = axes[row_idx, col_off + 0]
            _draw_background(ax, mooney_bg, "mooney", img_extent)
            _draw_reference_map(ax, mooney_map)
            _draw_fixations(ax, x_px, y_px)
            _panel_diagnostic(ax, x_px, y_px)

            # --- Col 1: Intact reference map + Mooney fixations ---
            ax = axes[row_idx, col_off + 1]
            _draw_background(ax, intact_bg, "disamb_intact", img_extent)
            _draw_reference_map(ax, intact_map)
            _draw_fixations(ax, x_px, y_px)
            _panel_score(ax, nss_intact)
            _panel_diagnostic(ax, intact_x, intact_y)

            # --- Col 2: Scrambled reference map + Mooney fixations ---
            ax = axes[row_idx, col_off + 2]
            _draw_background(ax, scrambled_bg, "disamb_not_intact", img_extent)
            _draw_reference_map(ax, scrambled_map)
            _draw_fixations(ax, x_px, y_px)
            _panel_score(ax, nss_scrambled)
            _panel_diagnostic(ax, scrambled_x, scrambled_y)

    # Lock every panel to the full canvas so the overlay lines up 1:1 with the maps
    triplet_titles = ["Mooney (where they looked)", "Intact ref map", "Scrambled ref map"]
    for row_idx, (row_label, _, _) in enumerate(rows):
        for col_idx in range(6):
            ax = axes[row_idx, col_idx]
            ax.set_xlim(0, IMAGE_WIDTH)
            ax.set_ylim(IMAGE_HEIGHT, 0)  # origin at top, y increases downward
            ax.set_aspect('auto')         # override imshow's aspect='equal' so axes fills its panel
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_idx == 0:
                # Extra pad on the disamb columns so the title clears the NSS label drawn
                # just above each panel (order top-to-bottom: title, NSS score, panel).
                within = col_idx % 3
                title_pad = 30 if within in (1, 2) else 12
                ax.set_title(triplet_titles[within], fontsize=14, fontweight='bold', pad=title_pad)
            if col_idx % 3 == 0:
                ax.set_ylabel(row_label, fontsize=15, fontweight='bold', labelpad=20)

    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    # Per-eye-group banner centred over each triplet
    for eye_idx, (_eye, eye_title) in enumerate(EYE_GROUPS):
        left_ax  = axes[0, eye_idx * 3]
        right_ax = axes[0, eye_idx * 3 + 2]
        x_center = (left_ax.get_position().x0 + right_ax.get_position().x1) / 2
        fig.text(x_center, 0.95, eye_title, ha='center', va='bottom',
                 fontsize=17, fontweight='bold')

    plt.suptitle(
        f"Image: {image_name}   |   ranked by unconscious_unaware NSS_diff (intact - scrambled) = {diff_uu:.3f}",
        fontsize=18, y=0.99
    )
    return fig

# ============================================================================
# MAIN
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results, fixations = load_data()
    if "dominant_eye" not in fixations.columns:
        raise SystemExit("❌ No 'dominant_eye' column in the parquet — rerun Stage 1 + "
                         "NSSExporter so the eye-split maps can be built.")
    ranked_images = rank_images_by_interaction(results)
    top = ranked_images[:N_TOP]

    print(f"\n🎨 Generating {len(top)} visualizations...")

    for rank, (diff, nss_intact, nss_scrambled, img_name) in enumerate(top, start=1):
        print(f"   [{rank:2d}/{len(top)}] Processing Image: {img_name}")
        fig = create_visualization(img_name, diff, fixations)
        output_file = OUTPUT_DIR / f"Rank_{rank:02d}_{img_name}.png"
        fig.savefig(output_file, dpi=DPI, bbox_inches='tight', transparent=False)
        plt.close(fig)

    print(f"\n✅ Done! All images saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
