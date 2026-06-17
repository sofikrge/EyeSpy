"""
NSS Top-20 Visualization Script

Generates 2x3 grid visualizations showing fixation heatmaps overlaid on:
- Intact disambiguators
- Mooney images  
- Scrambled disambiguators

For both Conscious and Unconscious sessions.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from scipy.fft import fft2, ifft2
import pandas as pd
from scipy.ndimage import gaussian_filter

# ============================================================================
# CONFIGURATION - Edit these paths and settings
# ============================================================================

# Data paths
BASE_PATH = Path("analysesresults/NSS")
STATS_PATH = BASE_PATH / "NSS_crossphase_descriptives.pkl"
MAPS_PATH = BASE_PATH / "FixMaps_full.pkl"
OUTPUT_DIR = Path("Figures/nss_separated_analyses/Top20_2x3")

FIXATIONS_PATH = Path("data/NSS_all_fixations_clean.parquet")
MASK_PPD = 48.55          # must match OldNSS.py's MASK_PPD
IMAGE_HEIGHT, IMAGE_WIDTH = 600, 800

# Stimulus directories
MOONEY_DIRS = [
    Path('/Users/sofiakarageorgiou/Documents/GitHub/Experiment Code/RUN_ME/Stimuli/ImageTrials_Experiment'),
    Path('/Users/sofiakarageorgiou/Documents/GitHub/Experiment Code/RUN_ME/Stimuli/ImageTrials_ExtraTrials')
]
DISAMB_DIRS = [
    Path('/Users/sofiakarageorgiou/Documents/GitHub/Experiment Code/RUN_ME/Stimuli/ImageDisamb_Experiment'),
    Path('/Users/sofiakarageorgiou/Documents/GitHub/Experiment Code/RUN_ME/Stimuli/ImageDisamb_ExtraTrials')
]

# Visualization settings
FIGURE_SIZE = (18, 11)
DPI = 150
HEATMAP_COLORMAP = "jet"
HEATMAP_ALPHA = 0.6
HEATMAP_THRESHOLD = 0  # Minimum density to show (0 = show all)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_image_file(img_name, directories):
    """Find image file in one of the provided directories."""
    filename = img_name if img_name.lower().endswith('.jpg') else f"{img_name}.jpg"
    for directory in directories:
        filepath = directory / filename
        if filepath.exists():
            return filepath
    return None


def scramble_image(image_path):
    """Create Fourier-scrambled version of image."""
    img = mpimg.imread(image_path)
    if img.dtype == np.uint8:
        img = img.astype(float) / 255.0
    
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    
    h, w, c = img.shape
    np.random.seed(42)
    random_phase = np.angle(fft2(np.random.rand(h, w)))
    random_phase[0, 0] = 0
    
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


def _round_half_away_from_zero(x):
    x = np.asarray(x, dtype=float)
    return (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(int)


def _deg_to_image_pixels(x_deg, y_deg, ppd, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    w_1based = _round_half_away_from_zero((width / 2.0) + x_deg * ppd)
    h_1based = _round_half_away_from_zero((height / 2.0) + y_deg * ppd)
    return h_1based, w_1based


def build_mooney_awareness_map(fixations_df, image_name, session, awareness_label, ppd=MASK_PPD):
    """Pool raw fixations for one image/session/awareness state into a fresh
    Gaussian-blurred density map (descriptive only, not the LOSO stats map)."""
    sub = fixations_df[
        (fixations_df["ImageName"].astype(str) == str(image_name)) &
        (fixations_df["session"].astype(str).str.upper() == str(session).upper()) &
        (fixations_df["image_type"] == "mooney_post_intact") &
        (fixations_df["awareness"] == awareness_label)
    ]
    if sub.empty:
        return None

    H, W = IMAGE_HEIGHT, IMAGE_WIDTH
    xdeg = pd.to_numeric(sub["x_deg_centered"], errors="coerce").to_numpy()
    ydeg = pd.to_numeric(sub["y_deg"], errors="coerce").to_numpy()
    ok = np.isfinite(xdeg) & np.isfinite(ydeg)
    if not ok.any():
        return None

    h1, w1 = _deg_to_image_pixels(xdeg[ok], ydeg[ok], ppd, width=W, height=H)
    inside = (h1 >= 1) & (h1 <= H) & (w1 >= 1) & (w1 <= W)
    if not inside.any():
        return None

    h0, w0 = h1[inside] - 1, w1[inside] - 1
    lin = (h0 * W + w0).astype(np.int64)
    counts = np.bincount(lin, minlength=H * W).reshape(H, W).astype(np.float32)

    sigma = float(ppd) / 2.0
    gaussian_filter(counts, sigma=sigma, mode="reflect", truncate=2.0, output=counts)
    return counts

def get_fixation_map(fixation_maps, image_name, session, image_type):
    """Retrieve fixation map for specific image/session/type."""
    for fmap in fixation_maps:
        if str(fmap['img']) != str(image_name):
            continue
        if str(fmap['condition']).upper() != str(session).upper():
            continue
        
        # Case 1: Mooney (Special strict check for post_intact)
        if image_type == "mooney":
             if fmap['image_type'] == 'mooney_post_intact':
                 return fmap['fixMapPerIm']
        
        # Case 2: Everything else (Direct match for disamb_intact / disamb_not_intact)
        elif fmap['image_type'] == image_type:
             return fmap['fixMapPerIm']

    return None


def load_data():
    """Load NSS statistics, fixation maps, and raw fixations (for awareness-split mooney maps)."""
    print(f"📂 Loading data...")
    with open(STATS_PATH, "rb") as f:
        stats = pickle.load(f)
    results = stats["data"] if isinstance(stats, dict) else stats
    
    with open(MAPS_PATH, "rb") as f:
        maps = pickle.load(f)["data"]

    fixations_df = pd.read_parquet(FIXATIONS_PATH)
    
    return results, maps, fixations_df


def rank_images_by_interaction(results):
    """Rank images by interaction effect: (C_diff - U_diff)"""
    ranked = []
    for entry in results['image']:
        if entry.get('condition') != 'C':
            continue
        
        img = entry['img']
        c_diff = entry.get('NSS_diff_img', 0)
        
        u_entry = next((e for e in results['image'] 
                       if e['img'] == img and e['condition'] == 'U'), None)
        u_diff = u_entry.get('NSS_diff_img', 0) if u_entry else 0
        
        interaction = c_diff - u_diff
        ranked.append((interaction, img))
    
    ranked.sort(reverse=True)
    return ranked


def create_visualization(rank, image_name, score, fixation_maps, fixations_df):
    """Create 2x3 visualization for one image."""
    # Load backgrounds
    mooney_bg = find_image_file(image_name, MOONEY_DIRS)
    intact_bg = find_image_file(image_name, DISAMB_DIRS)
    scrambled_bg = scramble_image(intact_bg) if intact_bg else None
    
    fig, axes = plt.subplots(3, 3, figsize=FIGURE_SIZE)
    
    rows = [
        ("Conscious (Aware)", "C", "conscious_aware"),
        ("Unconscious (Aware)", "U", "unconscious_aware"),
        ("Unconscious (Unaware)", "U", "unconscious_unaware"),
    ]
    columns = [
        ("Mooney", "mooney", mooney_bg),
        ("Intact Disambiguator", "disamb_intact", intact_bg),
        ("Scrambled Disambiguator", "disamb_not_intact", scrambled_bg)
    ]
    
    # Plot grid
    for row_idx, (row_name, session_code, awareness_label) in enumerate(rows):
        for col_idx, (col_title, img_type, background) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            
            # Plot background
            if background is not None:
                if isinstance(background, (str, Path)):
                    img = mpimg.imread(background)
                    cmap = 'gray' if 'mooney' in img_type else None
                    ax.imshow(img, origin='upper', cmap=cmap)
                else:
                    ax.imshow(background, origin='upper')
            
            # Overlay heatmap
            if img_type == "mooney":
                heatmap = build_mooney_awareness_map(fixations_df, image_name, session_code, awareness_label)
            else:
                heatmap = get_fixation_map(fixation_maps, image_name, session_code, img_type)

            if heatmap is not None:
                # heatmap = np.flipud(heatmap)  # Uncomment this ONLY if fixations look upside down
                norm_heatmap = heatmap / heatmap.max() if heatmap.max() > 0 else heatmap
                masked = np.ma.masked_where(norm_heatmap < HEATMAP_THRESHOLD, norm_heatmap)
                
                # Removed the extent parameter here too
                ax.imshow(masked, cmap=HEATMAP_COLORMAP, alpha=HEATMAP_ALPHA, origin='upper')
            
            # Styling
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            if row_idx == 0:
                ax.set_title(col_title, fontsize=16, fontweight='bold', pad=15)
            if col_idx == 0:
                ax.set_ylabel(row_name, fontsize=16, fontweight='bold', labelpad=25)
    
    fig.subplots_adjust(hspace=0.02, wspace=0.05)
    plt.suptitle(f"Image: {image_name} (Interaction: {score:.3f})", 
                 fontsize=22, y=0.98)
    return fig


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results, fixation_maps, fixations_df = load_data()
    ranked_images = rank_images_by_interaction(results)
    top_20 = ranked_images[:40]
    
    print(f"\n🎨 Generating {len(top_20)} visualizations...")
    for rank, (score, img_name) in enumerate(top_20, start=1):
        print(f"   [{rank:2d}/20] {img_name}")
        fig = create_visualization(rank, img_name, score, fixation_maps, fixations_df)
        output_file = OUTPUT_DIR / f"Rank_{rank:02d}_{img_name}.png"
        fig.savefig(output_file, dpi=DPI, bbox_inches='tight', transparent=True)
        plt.close(fig)
    
    print(f"\n✅ Done! Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()