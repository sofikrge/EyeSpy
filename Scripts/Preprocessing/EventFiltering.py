"""Dropping events and sessions that fail quality checks.

Two independent filters:
  - filter_and_report_validations  drops sessions whose calibration accuracy misses the
                                   thresholds in Settings.py
  - filter_events_blink_spatial    drops events overlapping a blink (skipped when blinks
                                   were interpolated instead) and events outside the
                                   image bounds or inside the central radius
"""

from Settings import FILTER_PALETTE
import polars as pl
import os
import copy
from pathlib import Path
import Scripts.Preprocessing.QualityPlots as plots

def filter_and_report_validations(dataset, data_quality_folder, avg_threshold, max_threshold):
    """Save a CSV summary of all validations, then filter out gaze samples
    that fall in 'bad' validation intervals (and plot sessions with bad data)."""

    os.makedirs(data_quality_folder, exist_ok=True)
    fileinfo = dataset.fileinfo['gaze']
    all_val_data = []

    print("\nChecking validation quality and filtering...")

    for i, g in enumerate(dataset.gaze):
        if g.validations is None:
            continue

        session_id = fileinfo.get_column('session_id')[i]
        participant_id = fileinfo.get_column('participant_id')[i]

        val_df = g.validations.sort("time")
        all_val_data.append( # for csv saving tagging with session and participant
            val_df.with_columns(session_id=pl.lit(session_id), participant_id=pl.lit(participant_id)))

        # Build interval boundaries: each validation marks the start of an interval,
        # and the last sample's time marks the end of the final one
        v_times = val_df["time"].to_list() + [g.samples["time"].max() + 1]
        
        # flag bad intervals based on thresholds
        is_bad = [(avg > avg_threshold or mx > max_threshold)
            for avg, mx in zip(val_df["accuracy_avg"], val_df["accuracy_max"])]

        # build good intervals based on bad flags: each bad validation marks the start of a bad interval, and the next validation marks its end
        good_intervals = [(v_times[j], v_times[j + 1]) for j, bad in enumerate(is_bad) if not bad]

        # only generate plot if at least one validation is bad
        if any(is_bad):
            plots.plot_validation_quality(g, v_times, is_bad, session_id, participant_id, data_quality_folder)

        # create boolean mask for samples to keep
        if good_intervals:
            keep_mask = pl.any_horizontal([
                (pl.col("time") >= start) & (pl.col("time") < end)
                for start, end in good_intervals
            ])
            g.samples = g.samples.filter(keep_mask)
        else:
            g.samples = g.samples.filter(pl.lit(False))

    # concat all validation data and save to csv for reporting
    if all_val_data:
        save_path = os.path.join(data_quality_folder, "validations.csv")
        pl.concat(all_val_data).write_csv(save_path)
def count_events(df):
    """Return (n_fixations, n_saccades) in a polars frame, in one pass."""
    # needed for filter_events_blink_spatial to track counts at each stage of filtering
    counts = df.group_by("name").agg(pl.len().alias("n"))
    counts_dict = dict(zip(counts["name"], counts["n"]))
    return counts_dict.get("fixation", 0), counts_dict.get("saccade", 0)
def filter_events_blink_spatial(dataset, raw_data_dir, buffer_fix, buffer_sac,
                                 hx, hy, center_radius_dg, data_quality_folder,
                                 debug=False, image_size_deg=None, filter_palette=None,
                                 skip_blink_filter=False):
    """
    Remove events that overlap blinks, fall outside the image, or fall inside the
    center radius. Tracks counts at each stage, saves a QC CSV, and (if debug)
    plots the filtering result per session.

    If `skip_blink_filter` is True (used when blinks were already interpolated
    upstream, so blink-spanning events are valid continuous fixations), the
    blink-overlap drop is skipped and only the spatial filtering is applied.
    """

    print("\nFiltering events (blinks and spatial bounds)...")
    os.makedirs(data_quality_folder, exist_ok=True)

    # Only keep a pre-filter copy if we're actually going to plot it
    events_prefilter = [copy.deepcopy(ev) for ev in dataset.events] if debug else None

    qc_data = []
    for i, ev in enumerate(dataset.events):
        p_id = dataset.fileinfo['gaze']['participant_id'][i]
        s_id = dataset.fileinfo['gaze']['session_id'][i]
        file_name = f"s_{s_id}_{p_id}.asc"

        # record starting fix sacc before any filtering
        df = ev.frame # working copy
        qc = {'participant_id': p_id, 'session_id': s_id}
        qc['fix_initial'], qc['sac_initial'] = count_events(df)

        # 1. Blink filtering
        blink_intervals = plots.parse_blink_intervals(os.path.join(raw_data_dir, file_name)) # extract all blinks
        qc['blinks_detected'] = len(blink_intervals)

        if blink_intervals and not skip_blink_filter: # if there are blinks, filter out events overlapping them + their buffers
            overlap_exprs = [
                (((pl.col("name") == "fixation") & (pl.col("onset") <= b_off + buffer_fix) & (pl.col("offset") >= b_on - buffer_fix)) |
                ((pl.col("name") == "saccade") & (pl.col("onset") <= b_off + buffer_sac) & (pl.col("offset") >= b_on - buffer_sac)))
                for b_on, b_off in blink_intervals
            ]
            df = df.filter(~pl.any_horizontal(overlap_exprs))



        # 2. Spatial filtering - add coordinates and distance from center
        df = df.with_columns(
            pl.col("location").list.get(0).alias("x"),
            pl.col("location").list.get(1).alias("y"),
        ).with_columns((pl.col("x") ** 2 + pl.col("y") ** 2).sqrt().alias("dist_from_center"))

        # 2a. Outside image bounds
        outside_mask = (pl.col("x").abs() > hx) | (pl.col("y").abs() > hy)
        fix_before, sac_before = count_events(df)
        df = df.filter(~outside_mask)
        fix_after, sac_after = count_events(df)
        qc['fix_outside'] = fix_before - fix_after
        qc['sac_outside'] = sac_before - sac_after

        # 2b. Inside center radius
        center_mask = pl.col("dist_from_center") <= center_radius_dg
        fix_before, sac_before = count_events(df)
        df = df.filter(~center_mask)
        fix_after, sac_after = count_events(df)
        qc['fix_center'] = fix_before - fix_after
        qc['sac_center'] = sac_before - sac_after

        qc['fix_final'], qc['sac_final'] = fix_after, sac_after

        ev.frame = df
        qc_data.append(qc)



    # Save QC report
    qc_df = pl.DataFrame(qc_data).select([
        'participant_id', 'session_id', 'blinks_detected',
        'fix_initial', 'fix_outside', 'fix_center', 'fix_final',
        'sac_initial',  'sac_outside', 'sac_center',  'sac_final'])
    qc_df.write_csv(os.path.join(data_quality_folder, "blink_spatial_filtering.csv"))

    if debug:
        plots.plot_fixation_filtering(
            events_list=events_prefilter, fileinfo=dataset.fileinfo,
            image_size_deg=image_size_deg, center_radius_dg=center_radius_dg,
            raw_data_dir=raw_data_dir, buffer_fix=buffer_fix,
            save_dir=Path(data_quality_folder) / "blinkspatial_filter",
            colors=filter_palette or FILTER_PALETTE)

    return qc_df
