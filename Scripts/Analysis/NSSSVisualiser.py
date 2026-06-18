"""
NSS Visualization Script
------------------------
This script generates a 3x3 grid of images for the Top 20 most interesting results.
It overlays "heatmaps" (where participants looked) on top of the actual experiment images.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from scipy.fft import fft2, ifft2

# ============================================================================
# CONFIGURATION - Settings you might need to change
# ============================================================================

# 1. Where to find the data and save the plots
BASE_PATH = Path("analysesresults/NSS")
STATS_PATH = BASE_PATH / "NSS_crossphase_descriptives.pkl"  # Contains the calculated scores
MAPS_PATH = BASE_PATH / "FixMaps_full.pkl"                  # Contains the pre-blurred heatmaps
OUTPUT_DIR = Path("Figures/nss_separated_analyses/Topppp20") # Where images will be saved

# 2. Screen and Eye-Tracking properties
MASK_PPD = 48.55          # Pixels Per Degree: How many pixels represent 1 degree of visual angle
IMAGE_HEIGHT = 600        # The total height of the eye-tracking capture area
IMAGE_WIDTH = 800         # The total width of the eye-tracking capture area

# 3. Where the original background images are stored on your computer
MOONEY_DIRS = [
    Path('/Users/sofiakarageorgiou/Documents/GitHub/Experiment Code/RUN_ME/Stimuli/ImageTrials_Experiment'),
    Path('/Users/sofiakarageorgiou/Documents/GitHub/Experiment Code/RUN_ME/Stimuli/ImageTrials_ExtraTrials')
]
DISAMB_DIRS = [
    Path('/Users/sofiakarageorgiou/Documents/GitHub/Experiment Code/RUN_ME/Stimuli/ImageDisamb_Experiment'),
    Path('/Users/sofiakarageorgiou/Documents/GitHub/Experiment Code/RUN_ME/Stimuli/ImageDisamb_ExtraTrials')
]

# 4. Visual settings for the final plot
FIGURE_SIZE = (18, 11)    # Width, Height of the final saved image
DPI = 150                 # Quality/Resolution of the final image
HEATMAP_COLORMAP = "jet"  # The color scheme for the heatmap (blue=low, red=high)
HEATMAP_ALPHA = 0.4       # Transparency of the heatmap (0 = invisible, 1 = solid)
HEATMAP_THRESHOLD = 0.05  # Hides heatmap values below this number to remove background noise/borders

# ============================================================================
# HELPER FUNCTIONS - Small tools to do specific tasks
# ============================================================================

def find_image_file(img_name, directories):
    """Looks through a list of folders to find the requested image file."""
    # Make sure the name ends with .jpg
    filename = img_name if img_name.lower().endswith('.jpg') else f"{img_name}.jpg"
    
    # Check each folder; return the exact path the moment we find it
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
    """Searches the loaded dataset for the specific pre-calculated heatmap we need."""
    for fmap in fixation_maps:
        # We need an exact match for the image name, the session (C/U), and the condition
        if str(fmap['img']) == str(image_name) and \
           str(fmap['condition']).upper() == str(session).upper() and \
           fmap['image_type'] == image_type:
            return fmap['fixMapPerIm']
    return None


def load_data():
    """Loads the statistics and the heatmaps from your computer's storage."""
    print(f"📂 Loading data...")
    with open(STATS_PATH, "rb") as f:
        stats = pickle.load(f)
    results = stats["data"] if isinstance(stats, dict) else stats
    
    with open(MAPS_PATH, "rb") as f:
        maps = pickle.load(f)["data"]
    
    return results, maps


def rank_images_by_interaction(results):
    """Finds which images had the biggest difference between Conscious and Unconscious viewing."""
    ranked = []
    
    for entry in results['image']:
        if entry.get('condition') != 'C':
            continue # Skip until we find a Conscious (C) entry
        
        img = entry['img']
        c_diff = entry.get('NSS_diff_img', 0)
        
        # Find the matching Unconscious (U) entry for this exact same image
        u_entry = next((e for e in results['image'] if e['img'] == img and e['condition'] == 'U'), None)
        u_diff = u_entry.get('NSS_diff_img', 0) if u_entry else 0
        
        # Calculate the interaction score (Difference of Differences)
        interaction = c_diff - u_diff
        ranked.append((interaction, img))
    
    # Sort them from highest score to lowest score
    ranked.sort(reverse=True)
    return ranked


def create_visualization(rank, image_name, score, fixation_maps):
    """Draws the actual 3x3 grid for a single image and saves it."""
    
    # 1. Find the background pictures
    mooney_bg = find_image_file(image_name, MOONEY_DIRS)
    intact_bg = find_image_file(image_name, DISAMB_DIRS)
    scrambled_bg = scramble_image(intact_bg) if intact_bg else None
    
    # 2. Calculate how to center the background picture inside the 800x600 eye-tracking canvas
    img_px_w = 9.99 * MASK_PPD # 9.99 degrees converted to pixels
    img_px_h = 7.50 * MASK_PPD # 7.50 degrees converted to pixels
    x_pad = (IMAGE_WIDTH - img_px_w) / 2
    y_pad = (IMAGE_HEIGHT - img_px_h) / 2
    # Define [left, right, bottom, top] boundaries for the background image
    img_extent = [x_pad, IMAGE_WIDTH - x_pad, IMAGE_HEIGHT - y_pad, y_pad]
    
    # 3. Create the blank 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=FIGURE_SIZE)
    
    # Define what goes on each row (Awareness States)
    rows = [
        ("Conscious (Aware)", "C", "conscious_aware"),
        ("Unconscious (Aware)", "U", "unconscious_aware"),
        ("Unconscious (Unaware)", "U", "unconscious_unaware"),
    ]
    # Define what goes on each column (Image Types)
    columns = [
        ("Mooney", "mooney", mooney_bg),
        ("Intact Disambiguator", "disamb_intact", intact_bg),
        ("Scrambled Disambiguator", "disamb_not_intact", scrambled_bg)
    ]
    
    # 4. Fill in the grid one box at a time
    for row_idx, (row_name, session_code, awareness_label) in enumerate(rows):
        for col_idx, (col_title, img_type, background) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            
            # --- Draw the Background ---
            if background is not None:
                if isinstance(background, (str, Path)):
                    img = mpimg.imread(background)
                    cmap = 'gray' if 'mooney' in img_type else None
                    # extent=img_extent locks the image to the center of the 800x600 space
                    ax.imshow(img, origin='upper', cmap=cmap, extent=img_extent)
                else:
                    ax.imshow(background, origin='upper', extent=img_extent)
            
            # --- Draw the Heatmap ---
            if img_type == "mooney":
                exact_mooney_type = f"mooney_post_intact_{awareness_label}"
                heatmap = get_fixation_map(fixation_maps, image_name, session_code, exact_mooney_type)
            else:
                heatmap = get_fixation_map(fixation_maps, image_name, session_code, img_type)
            
            if heatmap is not None:
                # Make the hottest spot '1.0' so colors are consistent across all subplots
                norm_heatmap = heatmap / heatmap.max() if heatmap.max() > 0 else heatmap
                
                # Make the empty/low-activity parts completely transparent so we can see the image
                masked = np.ma.masked_where(norm_heatmap < HEATMAP_THRESHOLD, norm_heatmap)
                
                # Plot the heatmap over the whole 800x600 area
                ax.imshow(masked, cmap=HEATMAP_COLORMAP, alpha=HEATMAP_ALPHA, origin='upper', extent=[0, IMAGE_WIDTH, IMAGE_HEIGHT, 0])
            
            # --- Make it look pretty ---
            ax.set_xticks([]) # Remove X axis numbers
            ax.set_yticks([]) # Remove Y axis numbers
            for spine in ax.spines.values():
                spine.set_visible(False) # Remove black borders around each subplot
            
            # Add titles only to the top row and left column
            if row_idx == 0:
                ax.set_title(col_title, fontsize=16, fontweight='bold', pad=15)
            if col_idx == 0:
                ax.set_ylabel(row_name, fontsize=16, fontweight='bold', labelpad=25)
    
    # 5. Final adjustments and save
    fig.subplots_adjust(hspace=0.02, wspace=0.05) # Squeeze the plots closer together
    plt.suptitle(f"Image: {image_name} (Interaction Score: {score:.3f})", fontsize=22, y=0.98)
    
    return fig


# ============================================================================
# MAIN SCRIPT - This is where the script actually starts running
# ============================================================================

def main():
    # Make sure the output folder exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get the data and figure out which 40 images to plot
    results, fixation_maps = load_data()
    ranked_images = rank_images_by_interaction(results)
    top_40 = ranked_images[:40] 
    
    print(f"\n🎨 Generating {len(top_40)} visualizations...")
    
    # Loop through the top images and build the plots
    for rank, (score, img_name) in enumerate(top_40, start=1):
        print(f"   [{rank:2d}/40] Processing Image: {img_name}")
        
        # Create the grid
        fig = create_visualization(rank, img_name, score, fixation_maps)
        
        # Save it to the hard drive
        output_file = OUTPUT_DIR / f"Rank_{rank:02d}_{img_name}.png"
        fig.savefig(output_file, dpi=DPI, bbox_inches='tight', transparent=False) # transparent=False prevents weird black backgrounds
        
        # Close the figure to free up computer memory
        plt.close(fig)
    
    print(f"\n✅ Done! All images saved to: {OUTPUT_DIR}")


# This tells Python to run the main() function only if we are running this specific file
if __name__ == "__main__":
    main()