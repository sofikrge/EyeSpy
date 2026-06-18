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
HEATMAP_ALPHA = 0.4
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


def get_fixation_map(fixation_maps, image_name, session, image_type):
    """Retrieve fixation map for specific image/session/type."""
    for fmap in fixation_maps:
        if str(fmap['img']) == str(image_name) and \
           str(fmap['condition']).upper() == str(session).upper() and \
           fmap['image_type'] == image_type:
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
    
    return results, maps


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


def create_visualization(rank, image_name, score, fixation_maps):
    """Create 2x3 visualization for one image."""
    # Load backgrounds
    mooney_bg = find_image_file(image_name, MOONEY_DIRS)
    intact_bg = find_image_file(image_name, DISAMB_DIRS)
    scrambled_bg = scramble_image(intact_bg) if intact_bg else None
    
    # Calculate true image pixel dimensions and padding for 800x600 canvas
    img_px_w = 9.99 * MASK_PPD
    img_px_h = 7.50 * MASK_PPD
    x_pad = (IMAGE_WIDTH - img_px_w) / 2
    y_pad = (IMAGE_HEIGHT - img_px_h) / 2
    img_extent = [x_pad, IMAGE_WIDTH - x_pad, IMAGE_HEIGHT - y_pad, y_pad]
    
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
                    ax.imshow(img, origin='upper', cmap=cmap, extent=img_extent)
                else:
                    ax.imshow(background, origin='upper', extent=img_extent)
            
            # Overlay heatmap
            if img_type == "mooney":
                # Construct the exact string saved in your NSS cache
                exact_mooney_type = f"mooney_post_intact_{awareness_label}"
                heatmap = get_fixation_map(fixation_maps, image_name, session_code, exact_mooney_type)
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
    
    results, fixation_maps = load_data()
    ranked_images = rank_images_by_interaction(results)
    top_20 = ranked_images[:40]
    
    print(f"\n🎨 Generating {len(top_20)} visualizations...")
    for rank, (score, img_name) in enumerate(top_20, start=1):
        print(f"   [{rank:2d}/20] {img_name}")
        fig = create_visualization(rank, img_name, score, fixation_maps)
        output_file = OUTPUT_DIR / f"Rank_{rank:02d}_{img_name}.png"
        fig.savefig(output_file, dpi=DPI, bbox_inches='tight', transparent=True)
        plt.close(fig)
    
    print(f"\n✅ Done! Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()