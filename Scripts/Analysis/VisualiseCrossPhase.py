# VisualiseCrossPhase

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from PIL import Image
import pickle
from scipy.ndimage import gaussian_filter

# === CONFIG ===
FIX_FILE = Path("data/NSS_all_fixations_clean.parquet")
FIXMAPS_CACHE = Path("analysesresults/NSS/FixMaps_full.pkl")
OUTPUT_DIR = Path("Figures/nss_visualisations")

MOONEY_DIRS = [
    Path('/Users/sofiakarageorgiou/Documents/GitHub/Experiment Code/RUN_ME/Stimuli/ImageTrials_Experiment'),
    Path('/Users/sofiakarageorgiou/Documents/GitHub/Experiment Code/RUN_ME/Stimuli/ImageTrials_ExtraTrials')
]
DISAMB_DIRS = [
    Path('/Users/sofiakarageorgiou/Documents/GitHub/Experiment Code/RUN_ME/Stimuli/ImageDisamb_Experiment'),
    Path('/Users/sofiakarageorgiou/Documents/GitHub/Experiment Code/RUN_ME/Stimuli/ImageDisamb_ExtraTrials')
]

IMAGE_WIDTH, IMAGE_HEIGHT = 800, 600
MASK_PPD = 48.55

# === HELPERS ===
def z_normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Z-normalizes the heatmap (matches NSS.py logic)."""
    mu = heatmap.mean()
    sd = heatmap.std(ddof=0) # Population SD as used in NSS.py
    if sd == 0 or not np.isfinite(sd):
        return np.zeros_like(heatmap)
    return (heatmap - mu) / sd

def find_image_file(image_name: str, search_dirs: list[Path], is_mooney: bool, phase: str = "") -> Path | None:
    """Finds the actual image file, handling potential suffixes for intact/scrambled."""
    base_name = image_name.split(".") # Extracts only '1002'
    search_names = [f"{base_name}.png", f"{base_name}.jpg", image_name]

    for directory in search_dirs:
        if not directory.exists(): continue
        for name in search_names:
            target = directory / name
            if target.exists(): return target
    return None

def round_half_away_from_zero(x):
    x = np.asarray(x, dtype=float)
    return (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(int)

def deg_to_pixels(x_deg, y_deg, ppd,
                  width=IMAGE_WIDTH,
                  height=IMAGE_HEIGHT):

    w_1based = round_half_away_from_zero(
        (width / 2.0) + x_deg * ppd
    )

    h_1based = round_half_away_from_zero(
        (height / 2.0) + y_deg * ppd
    )

    return h_1based, w_1based

def generate_heatmap_from_coords(df: pd.DataFrame, ppd=MASK_PPD) -> np.ndarray:
    """Generates a blurred heatmap directly from raw fixations (for Mooney Aware/Unaware splits)."""
    heatmap = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
    if df.empty: return heatmap

    xdeg = pd.to_numeric(df["x_deg_centered"], errors="coerce").to_numpy()
    ydeg = pd.to_numeric(df["y_deg"], errors="coerce").to_numpy()
    ok = np.isfinite(xdeg) & np.isfinite(ydeg)
    
    if ok.any():
        h1, w1 = deg_to_pixels(xdeg[ok], ydeg[ok], ppd)
        inside = (h1 >= 1) & (h1 <= IMAGE_HEIGHT) & (w1 >= 1) & (w1 <= IMAGE_WIDTH)
        h1, w1 = h1[inside] - 1, w1[inside] - 1
        
        lin = (h1 * IMAGE_WIDTH + w1).astype(np.int64)
        counts = np.bincount(lin, minlength=IMAGE_HEIGHT * IMAGE_WIDTH).reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
        heatmap += counts.astype(np.float32)
        
        sigma = ppd / 2.0
        gaussian_filter(heatmap, sigma=sigma, mode="reflect", truncate=2.0, output=heatmap)
        
    return heatmap

def overlay_heatmap(ax, bg_path: Path | None, heatmap: np.ndarray, title: str):
    if bg_path and bg_path.exists():
        img = Image.open(bg_path).convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        ax.imshow(img)
    else:
        ax.set_facecolor('black')
        ax.text(IMAGE_WIDTH/2, IMAGE_HEIGHT/2, "Image Not Found", color='white', ha='center')

    # Apply NSS-style Z-normalization
    z_map = z_normalize_heatmap(heatmap)
    
    # Rescale to for display purposes (Min-Max scaling of the Z-scores)
    # This keeps the visualization readable while reflecting the Z-scored distribution
    if z_map.max() > z_map.min():
        z_display = (z_map - z_map.min()) / (z_map.max() - z_map.min())
        # Mask out low values (e.g., < 0.2)
        heatmap_masked = np.ma.masked_where(z_display < 0.2, z_display)
        ax.imshow(heatmap_masked, cmap='jet', alpha=0.6, extent=[0, IMAGE_WIDTH, IMAGE_HEIGHT, 0])
    
    ax.set_title(title, fontsize=10, pad=10)
    ax.axis('off')

# === MAIN SCRIPT ===
def plot_cross_phase_overlay(image_name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    df = pd.read_parquet(FIX_FILE)
    df_img = df[df["ImageName"] == image_name]
    
    if df_img.empty:
        print(f"No fixations found for image: {image_name}")
        return

    with open(FIXMAPS_CACHE, "rb") as f:
        fixmaps = pickle.load(f)["data"]
    
    # Fast lookup for Session Reference Maps
    fm_index = {(fm["condition"], fm["image_type"]): fm["fixMapPerIm"] 
                for fm in fixmaps if fm["img"] == image_name}

    # 2. Find Base Images
    path_mooney = find_image_file(image_name, MOONEY_DIRS, is_mooney=True)
    path_intact = find_image_file(image_name, DISAMB_DIRS, is_mooney=False, phase="intact")
    path_scrambled = None  # Scrambled images are generated on the fly, no file exists

    # 3. Setup Figure Grid
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f"Cross-Phase Fixation Overlays: {image_name}", fontsize=16, fontweight='bold')
    
    # 12-column grid allows clean math for both 3-col (span 4) and 4-col (span 3) layouts
    gs = GridSpec(2, 12, figure=fig, hspace=0.3, wspace=0.3)

    # Insert this check to see what the data actually contains
    print(f"DEBUG: Unique image_types in df_img: {df_img['image_type'].unique()}")
    print(f"DEBUG: Unique awareness in df_img: {df_img['awareness'].unique()}")

    # --- ROW 1: CONSCIOUS (3 Columns -> spans 0:4, 4:8, 8:12) ---
    df_c = df_img[df_img["session"] == "C"]
    # Temporarily remove the awareness filter to see if anything plots
    df_c_mooney = df_c[df_c["image_type"].str.contains("mooney", na=False)]
    hm_c_mooney_aware = generate_heatmap_from_coords(df_c_mooney[df_c_mooney["awareness"] == "conscious_aware"])
    # (And add a Conscious Unaware heatmap if that data exists)
    hm_c_intact = fm_index.get(("C", "disamb_intact"), np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH)))
    hm_c_scrambled = fm_index.get(("C", "disamb_not_intact"), np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH)))

    ax1 = fig.add_subplot(gs[0, 0:4])
    overlay_heatmap(ax1, path_mooney, hm_c_mooney_aware, "Conscious: Mooney (Aware)")

    ax2 = fig.add_subplot(gs[0, 4:8])
    overlay_heatmap(ax2, path_intact, hm_c_intact, "Conscious: Intact (Session Avg)")

    ax3 = fig.add_subplot(gs[0, 8:12])
    overlay_heatmap(ax3, path_scrambled, hm_c_scrambled, "Conscious: Scrambled (Session Avg)")


    # --- ROW 2: UNCONSCIOUS (4 Columns -> spans 0:3, 3:6, 6:9, 9:12) ---
    df_u = df_img[df_img["session"] == "U"]
    df_mooney = df_u[df_u["image_type"].str.contains("mooney", na=False)]

    print("\n----- MOONEY DATA -----")
    print(df_mooney["awareness"].value_counts())

    aware_df = df_mooney[
        df_mooney["awareness"] == "unconscious_aware"
    ]

    unaware_df = df_mooney[
        df_mooney["awareness"] == "unconscious_unaware"
    ]

    print("Aware rows:", len(aware_df))
    print("Unaware rows:", len(unaware_df))

    hm_u_mooney_aware = generate_heatmap_from_coords(aware_df)
    hm_u_mooney_unaware = generate_heatmap_from_coords(unaware_df)

    print("Aware heatmap max:", hm_u_mooney_aware.max())
    print("Unaware heatmap max:", hm_u_mooney_unaware.max())

    hm_u_intact = fm_index.get(("U", "disamb_intact"), np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH)))
    hm_u_scrambled = fm_index.get(("U", "disamb_not_intact"), np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH)))

    ax4 = fig.add_subplot(gs[1, 0:3])
    overlay_heatmap(ax4, path_mooney, hm_u_mooney_aware, "Unconscious: Mooney (Aware)")

    ax5 = fig.add_subplot(gs[1, 3:6])
    overlay_heatmap(ax5, path_mooney, hm_u_mooney_unaware, "Unconscious: Mooney (Unaware)")

    ax6 = fig.add_subplot(gs[1, 6:9])
    overlay_heatmap(ax6, path_intact, hm_u_intact, "Unconscious: Intact (Session Avg)")

    ax7 = fig.add_subplot(gs[1, 9:12])
    overlay_heatmap(ax7, path_scrambled, hm_u_scrambled, "Unconscious: Scrambled (Session Avg)")

    # Save and cleanup
    out_path = OUTPUT_DIR / f"Overlay_{image_name}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved visualization for {image_name} -> {out_path}")
    plt.show()

if __name__ == "__main__":
    # Example usage:
    # Replace "Image_01" with a real image name from your dataframe
    plot_cross_phase_overlay("1009.jpg")