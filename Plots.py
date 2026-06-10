#%% Imports
import os
import polars as pl
import matplotlib.pyplot as plt

def plot_validation_quality(g, v_times, is_bad, s_id, p_id, folder):
    """Plot gaze x-position with validation intervals, highlighting bad ones."""
    plt.figure(figsize=(15, 5))
    plt.plot(g.samples["time"], g.samples["position"].list.get(0),
             color='grey', alpha=0.6, linewidth=0.5)

    for j, bad in enumerate(is_bad):
        plt.axvline(x=v_times[j], color='#b3cde3', linestyle='--', alpha=0.8)
        if bad:
            plt.axvspan(v_times[j], v_times[j + 1], color='#88419d', alpha=0.2)

    plt.title(f"Validations Data Quality: Participant {p_id} (Session {s_id})")
    plt.xlabel("Time (ms)")
    plt.ylabel("X Position (Visual Degrees)")
    plt.savefig(os.path.join(folder, f"validations_plot_{s_id}_{p_id}.svg"))
    plt.close()