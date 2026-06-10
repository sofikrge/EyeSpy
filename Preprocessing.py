from Settings import EYE_OFFSET
import polars as pl
import matplotlib.pyplot as plt
import os

def shift_gaze_offset(dataset, eye_offset=EYE_OFFSET):
    """Shift gaze coordinates by a specified offset for left and right eyes."""
    
    for g in dataset.gaze:
        md = getattr(g, "metadata", None) or getattr(g, "_metadata", None) or {}
        te = md.get("tracked_eye")
        
        # Determine eye and offset
        eye = "left" if str(te or "R").strip().upper() in ("L", "LEFT") else "right"
        off = eye_offset.get(eye, 0.0)
        
        # Apply offset
        g.samples = g.samples.with_columns(
            position = pl.concat_list([
                pl.col("position").list.get(0) + off,
                pl.col("position").list.get(1)]))
        
    return dataset

def filter_and_report_validations(dataset, data_quality_folder,
                                   avg_threshold, max_threshold):
    """Save a CSV summary of all validations, then filter out gaze samples
    that fall in 'bad' validation intervals (and plot sessions with bad data)."""

    os.makedirs(data_quality_folder, exist_ok=True)
    fileinfo = dataset.fileinfo['gaze']
    all_val_data = []

    print("\nChecking validation quality and filtering...")

    for i, g in enumerate(dataset.gaze):
        if g.validations is None:
            continue

        s_id = fileinfo.get_column('session_id')[i]
        p_id = fileinfo.get_column('participant_id')[i]

        val_df = g.validations.sort("time")
        all_val_data.append(
            val_df.with_columns(session_id=pl.lit(s_id), participant_id=pl.lit(p_id))
        )

        # Build interval boundaries: each validation marks the start of an interval,
        # and the last sample's time marks the end of the final one.
        v_times = val_df["time"].to_list() + [g.samples["time"].max() + 1]
        is_bad = [
            (avg > avg_threshold or mx > max_threshold)
            for avg, mx in zip(val_df["accuracy_avg"], val_df["accuracy_max"])
        ]
        good_intervals = [
            (v_times[j], v_times[j + 1]) for j, bad in enumerate(is_bad) if not bad
        ]

        if any(is_bad):
            _plot_validation_quality(g, v_times, is_bad, s_id, p_id, data_quality_folder)

        if good_intervals:
            keep_mask = pl.any_horizontal([
                (pl.col("time") >= start) & (pl.col("time") < end)
                for start, end in good_intervals
            ])
            g.samples = g.samples.filter(keep_mask)
        else:
            g.samples = g.samples.filter(pl.lit(False))

    if all_val_data:
        save_path = os.path.join(data_quality_folder, "validations.csv")
        pl.concat(all_val_data).write_csv(save_path)

def _plot_validation_quality(g, v_times, is_bad, s_id, p_id, folder):
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