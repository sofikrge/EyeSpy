# Plots.py

#%% Imports
import os
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as patches
import random
from bisect import bisect_left
from FileParsing import parse_blink_intervals, parse_trials_from_asc

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

def plot_fixation_filtering(events_list, fileinfo, image_size_deg, center_radius_dg,
                             raw_data_dir, buffer_fix, save_dir, colors=None):
    """Plot fixations colored by filtering reason using sequential filtering logic."""

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if colors is None:
        colors = ['#edf8fb', '#b3cde3', '#8c96c6', '#88419d']
    color_map = {'blink': colors[1], 'outside': colors[2], 'center': colors[2], 'kept': colors[3]}

    hx, hy = image_size_deg[0] / 2, image_size_deg[1] / 2

    for i, ev in enumerate(events_list):
        p_id = fileinfo['gaze']['participant_id'][i]
        s_id = fileinfo['gaze']['session_id'][i]
        file_name = f"s_{s_id}_{p_id}.asc"

        df = ev.frame.filter(pl.col("name") == "fixation")
        if df.is_empty():
            continue

        try:
            blink_intervals = parse_blink_intervals(os.path.join(raw_data_dir, file_name))
        except FileNotFoundError:
            print(f"Warning: Could not find {file_name} for blink parsing.")
            blink_intervals = []

        df = _label_fixations(df, blink_intervals, buffer_fix, hx, hy, center_radius_dg)
        _plot_fixation_session(df, p_id, s_id, hx, hy, center_radius_dg, color_map, save_dir)


def _label_fixations(df, blink_intervals, buffer_fix, hx, hy, center_radius_dg):
    """Add x, y, r columns and a 'reason' column via sequential filtering logic."""

    overlap_expr = pl.lit(False)
    for b_on, b_off in blink_intervals:
        overlap_expr |= (
            (pl.col("onset") <= b_off + buffer_fix) &
            (pl.col("offset") >= b_on - buffer_fix)
        )

    df = df.with_columns(
        overlap_expr.alias("blink_overlap"),
        pl.col("location").list.get(0).alias("x"),
        pl.col("location").list.get(1).alias("y"),
    ).with_columns((pl.col("x") ** 2 + pl.col("y") ** 2).sqrt().alias("r"))

    return df.with_columns(
        pl.when(pl.col("blink_overlap")).then(pl.lit("blink"))
        .when((pl.col("x").abs() > hx) | (pl.col("y").abs() > hy)).then(pl.lit("outside"))
        .when(pl.col("r") <= center_radius_dg).then(pl.lit("center"))
        .otherwise(pl.lit("kept"))
        .alias("reason")
    )


def _plot_fixation_session(df, p_id, s_id, hx, hy, center_radius_dg, color_map, save_dir):
    """Draw and save the fixation scatter plot for one session."""

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot in order so 'kept' ends up on top
    for reason in ['blink', 'outside', 'center', 'kept']:
        subset = df.filter(pl.col("reason") == reason)
        if not subset.is_empty():
            ax.scatter(subset["x"], subset["y"], c=color_map[reason], s=15,
                       alpha=0.8, edgecolor='none', label=reason.capitalize())

    # Reference shapes
    ax.add_patch(patches.Rectangle((-hx, -hy), 2 * hx, 2 * hy,
                                    fill=False, edgecolor="black", lw=1))
    ax.add_patch(patches.Circle((0, 0), center_radius_dg,
                                 fill=False, edgecolor="gray", lw=1, linestyle=":"))

    # Counts box
    counts = df.group_by("reason").agg(pl.len().alias("n"))
    count_dict = dict(zip(counts["reason"], counts["n"]))
    legend_text = "\n".join(
        f"{key.capitalize()}: {count_dict[key]}"
        for key in ['kept', 'center', 'outside', 'blink'] if key in count_dict
    )
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.9))

    ax.set_xlabel("X position (deg)")
    ax.set_ylabel("Y position (deg)")
    ax.set_title(f"Fixation Filtering: s_{s_id}_{p_id}")
    ax.set_aspect('equal')
    ax.legend(loc='upper right')

    fig.savefig(save_dir / f"s_{s_id}_{p_id}_filtering.svg", dpi=200, bbox_inches='tight')
    plt.close(fig)

def _trial_phase_windows(trial_meta):
    """Extract phase window times and condition from a single trial_meta row."""
    return {
        't_start': trial_meta["trial_start"][0],
        'd_start': trial_meta["disambig_start"][0],
        'd_end':   trial_meta["disambig_end"][0],
        'm_start': trial_meta["mooney_start"][0],
        'm_end':   trial_meta["mooney_end"][0],
        'condition': trial_meta["condition"][0],
    }


def _plot_trial_alignment(trial_events, windows, palette_map, p_id, s_id, t_num, save_dir):
    """Plot ground-truth phase windows vs. assigned phase labels for one trial."""

    fig, ax = plt.subplots(figsize=(12, 4))

    # Ground truth windows
    ax.axvspan(windows['m_start'], windows['m_end'],
               color=palette_map['mooney'], alpha=0.2, label='TRUE Mooney Interval')
    ax.axvspan(windows['d_start'], windows['d_end'],
               color=palette_map['disambiguation'], alpha=0.2, label='TRUE Disambig Interval')

    # Assigned phase labels
    for phase in ['inter_stimulus', 'mooney', 'disambiguation']:
        subset = trial_events.filter(pl.col("phase") == phase)
        if not subset.is_empty():
            ax.scatter(subset["onset"], [1] * len(subset),
                       color=palette_map.get(phase, 'black'),
                       s=50, edgecolors='white', zorder=10,
                       label=f'Assigned: {phase}')

    ax.set_yticks([])
    ax.set_xlabel("Time (ms)")
    ax.set_title(f"Alignment Check: Participant {p_id}, Trial {t_num} ({windows['condition'].upper()})")
    ax.legend(loc='upper right', fontsize='small')

    pad = 500
    ax.set_xlim(windows['t_start'] - pad, windows['m_end'] + pad)

    save_path = save_dir / f"alignment_check_{s_id}_{p_id}_trial{t_num}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"   Saved check plot: {save_path}")


def plot_phase_alignment_check(events_list, fileinfo, raw_data_dir, save_dir,
                                labels, patterns, n_trials=3, colors=None):
    """Visual check that events are correctly assigned to phases, for a random
    sample of sessions and trials."""

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if colors is None:
        colors = ['#b3cde3', '#8c96c6', '#88419d']  # [inter, mooney, disambig]
    palette_map = {'inter_stimulus': colors[0], 'mooney': colors[1], 'disambiguation': colors[2]}

    selected_indices = random.sample(range(len(events_list)), min(5, len(events_list)))

    for i in selected_indices:
        ev = events_list[i]
        p_id = fileinfo['gaze']['participant_id'][i]
        s_id = fileinfo['gaze']['session_id'][i]
        filepath = os.path.join(raw_data_dir, f"s_{s_id}_{p_id}.asc")

        df_trials = parse_trials_from_asc(filepath, labels=labels, patterns=patterns)
        df_events = ev.frame

        all_trials = df_trials["trial_number"].unique().to_list()
        selected_trials = sorted(random.sample(all_trials, min(n_trials, len(all_trials))))

        for t_num in selected_trials:
            trial_meta = df_trials.filter(pl.col("trial_number") == t_num)
            if trial_meta.is_empty():
                continue

            windows = _trial_phase_windows(trial_meta)
            if windows['d_start'] is None or windows['m_start'] is None:
                print(f"Skipping alignment plot for Trial {t_num} (Incomplete data)")
                continue

            trial_events = df_events.filter(pl.col("trial_number") == t_num)
            _plot_trial_alignment(trial_events, windows, palette_map, p_id, s_id, t_num, save_dir)