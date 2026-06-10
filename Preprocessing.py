from Settings import EYE_OFFSET
import polars as pl
import matplotlib.pyplot as plt
import os
import Plots as plots
import copy
from pathlib import Path

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
            plots.plot_validation_quality(g, v_times, is_bad, s_id, p_id, data_quality_folder)

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

def parse_blink_intervals(file_path):
    """Extract (onset, offset) tuples for each EBLINK line in an .asc file."""
    intervals = []
    with open(file_path) as f:
        for line in f:
            if line.startswith("EBLINK"):
                parts = line.split()
                intervals.append((int(parts[2]), int(parts[3])))
    return intervals


def count_events(df):
    """Return (n_fixations, n_saccades) in a polars frame, in one pass."""
    counts = df.group_by("name").agg(pl.len().alias("n"))
    counts_dict = dict(zip(counts["name"], counts["n"]))
    return counts_dict.get("fixation", 0), counts_dict.get("saccade", 0)


def filter_events_blink_spatial(dataset, raw_data_dir, buffer_fix, buffer_sac,
                                 hx, hy, center_radius_dg, data_quality_folder,
                                 debug=False, image_size_deg=None, filter_palette=None):
    """
    Remove events that overlap blinks, fall outside the image, or fall inside the
    center radius. Tracks counts at each stage, saves a QC CSV, and (if debug)
    plots the filtering result per session.
    """

    print("\nBlink and Spatial Filtering of Events...")
    os.makedirs(data_quality_folder, exist_ok=True)

    # Only keep a pre-filter copy if we're actually going to plot it
    events_prefilter = [copy.deepcopy(ev) for ev in dataset.events] if debug else None

    qc_data = []
    for i, ev in enumerate(dataset.events):
        p_id = dataset.fileinfo['gaze']['participant_id'][i]
        s_id = dataset.fileinfo['gaze']['session_id'][i]
        file_name = f"s_{s_id}_{p_id}.asc"

        df = ev.frame
        qc = {'participant_id': p_id, 'session_id': s_id}
        qc['fix_initial'], qc['sac_initial'] = count_events(df)

        # 1. Blink filtering
        blink_intervals = parse_blink_intervals(os.path.join(raw_data_dir, file_name))
        qc['blinks_detected'] = len(blink_intervals)

        if blink_intervals:
            overlap_expr = pl.lit(False)
            for b_on, b_off in blink_intervals:
                overlap_expr |= (
                    ((pl.col("name") == "fixation") &
                     (pl.col("onset") <= b_off + buffer_fix) &
                     (pl.col("offset") >= b_on - buffer_fix)) |
                    ((pl.col("name") == "saccade") &
                     (pl.col("onset") <= b_off + buffer_sac) &
                     (pl.col("offset") >= b_on - buffer_sac))
                )
            df = df.filter(~overlap_expr)

        # 2. Spatial filtering - add coordinates and distance from center
        df = df.with_columns(
            pl.col("location").list.get(0).alias("x"),
            pl.col("location").list.get(1).alias("y"),
        ).with_columns((pl.col("x") ** 2 + pl.col("y") ** 2).sqrt().alias("r"))

        # 2a. Outside image bounds
        outside_mask = (pl.col("x").abs() > hx) | (pl.col("y").abs() > hy)
        qc['fix_outside'], qc['sac_outside'] = count_events(df.filter(outside_mask))
        df = df.filter(~outside_mask)

        # 2b. Inside center radius
        center_mask = pl.col("r") <= center_radius_dg
        qc['fix_center'], qc['sac_center'] = count_events(df.filter(center_mask))
        df = df.filter(~center_mask)

        qc['fix_final'], qc['sac_final'] = count_events(df)

        ev.frame = df.rename({"r": "dist_from_center"})
        qc_data.append(qc)

    # Save QC report
    qc_df = pl.DataFrame(qc_data).select([
        'participant_id', 'session_id', 'blinks_detected',
        'fix_initial', 'sac_initial', 'fix_outside', 'sac_outside',
        'fix_center', 'sac_center', 'fix_final', 'sac_final'
    ])
    qc_df.write_csv(os.path.join(data_quality_folder, "blink_spatial_filtering.csv"))

    if debug:
        plot_fixation_filtering(
            events_list=events_prefilter, fileinfo=dataset.fileinfo,
            image_size_deg=image_size_deg, center_radius_dg=center_radius_dg,
            raw_data_dir=raw_data_dir, buffer_fix=buffer_fix,
            save_dir=Path(data_quality_folder) / "blinkspatial_filter",
            colors=filter_palette or ['#edf8fb', '#b3cde3', '#8c96c6', '#88419d']
        )

    return qc_df