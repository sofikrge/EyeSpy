# Preprocessing.py

from Settings import EYE_OFFSET, FILTER_PALETTE, PHASE_PALETTE, MAT_FIELD_MAP
import polars as pl
import os
import Scripts.Preprocessing.Plots as plots
import copy
from pathlib import Path
import numpy as np
import scipy.io as sio

def shift_gaze_offset(dataset, eye_offset=EYE_OFFSET):
    """Shift gaze coordinates by a specified offset for left and right eyes."""
    
    for g in dataset.gaze:
        md = getattr(g, "metadata", None) or getattr(g, "_metadata", None) or {}
        te = md.get("tracked_eye")
        
        # Determine eye and offset
        eye = "left" if str(te or "R").strip().upper() in ("L", "LEFT") else "right" # if empty default to right, normalise naming
        off = eye_offset.get(eye, 0.0) # look up offset for this eye, default to 0 if not found
        
        # Apply offset
        g.samples = g.samples.with_columns(
            position = pl.concat_list([
                pl.col("position").list.get(0) + off,
                pl.col("position").list.get(1)]))
        
    return dataset

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

        # record starting fix sacc before any filtering
        df = ev.frame # working copy
        qc = {'participant_id': p_id, 'session_id': s_id}
        qc['fix_initial'], qc['sac_initial'] = count_events(df)

        # 1. Blink filtering
        blink_intervals = plots.parse_blink_intervals(os.path.join(raw_data_dir, file_name)) # extract all blinks
        qc['blinks_detected'] = len(blink_intervals)

        if blink_intervals: # if there are blinks, filter out events overlapping them + their buffers
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

def _find_expdata_struct(mat):
    """Find the 'expdata' key in a loaded .mat dict, case-insensitively."""
    for key, value in mat.items():
        if key.lower() == 'expdata':
            return value
    return None

def _is_ghost_trial(t):
    """A 'ghost' trial is an empty placeholder row with no TrialNum data. This function detects that case so it can be skipped"""
    t_num = getattr(t, 'TrialNum', None) # access matlab truct to be loaded viea scipy
    return isinstance(t_num, np.ndarray) and t_num.size == 0

def load_behavioural_from_mat(mat_path, section_map):
    """
    Parse a behavioural .mat file into a DataFrame, matching trials by row
    order (row index i -> trial_number i+1), trusting that the MAT and ASC
    files record trials in the same sequence.
    """

    if not os.path.exists(mat_path):
        print(f"   ⚠️ MAT File not found: {mat_path}")
        return pl.DataFrame()

    try:
        mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False) # turn into flat scalar struct, remove dimensions, turn to python object
        expdata = _find_expdata_struct(mat)
        if expdata is None:
            print(f"   ❌ Key 'expdata' (or similar) not found in {os.path.basename(mat_path)}")
            return pl.DataFrame()
    except Exception as e:
        print(f"   ❌ Error reading MAT structure: {e}")
        return pl.DataFrame()

    beh_rows = [] # loop over eachv section (e.g. practice, block1) and extract trials, order matches ASC
    for section_field, block_name in section_map.items():
        if not hasattr(expdata, section_field):
            continue

        trials_struct = getattr(expdata, section_field)
        if not isinstance(trials_struct, np.ndarray):
            trials_struct = [trials_struct]

        for i, t in enumerate(trials_struct):
            if _is_ghost_trial(t):
                continue

            row = {'block_type': block_name, 'trial_number': i + 1}
            row.update({out: getattr(t, src, None) for src, out in MAT_FIELD_MAP.items()})
            beh_rows.append(row)

    df = pl.DataFrame(beh_rows, infer_schema_length=None)
    if not df.is_empty():
        print(f"   ✅ Loaded {df.height} trials from {os.path.basename(mat_path)}")
    return df

def assign_trial_metadata_and_phases(dataset, raw_data_dir, behavioural_dir, events_out_dir,
                                      trial_labels, asc_patterns, section_to_block,
                                      debug=False, data_quality_folder=None, phase_palette=None):
    """
    For each session: parse trial timings from the .asc file, merge in behavioural
    data from the .mat file, join onto events, assign a phase label per event,
    and save the result. If debug, also run the phase-alignment QC plots.
    """

    os.makedirs(events_out_dir, exist_ok=True)

    for i, ev in enumerate(dataset.events):
        s_id = dataset.fileinfo['gaze']['session_id'][i]
        p_id = dataset.fileinfo['gaze']['participant_id'][i]
        asc_path = os.path.join(raw_data_dir, f"s_{s_id}_{p_id}.asc")
        mat_path = os.path.join(behavioural_dir, f"expdata_{s_id}_{p_id}.mat")
        csv_name = f"s_{s_id}_{p_id}.csv"

        # Parse trial timings from the ASC, and behavioural data from the MAT
        df_trials_asc = plots.parse_trials_from_asc(asc_path, labels=trial_labels, patterns=asc_patterns)
        df_beh = load_behavioural_from_mat(mat_path, section_to_block)

        # Merge ASC + MAT
        if not df_beh.is_empty():
            df_trials_combined = df_trials_asc.join(
                df_beh, on=['block_type', 'trial_number'], how='left'
            )
        else:
            df_trials_combined = df_trials_asc

        # Check for mismatches
        n_missing = df_trials_combined.filter(pl.col("ImageName").is_null()).height
        if n_missing > 0:
            print(f"⚠️ WARNING: {n_missing} trials missing behavioral data after merge!")
            print("   This may indicate ordinal mismatch between ASC and MAT files.")

        if not df_beh.is_empty():
            total_trials = df_trials_combined.height
            matched_trials = df_trials_combined.filter(pl.col("ImageName").is_not_null()).height
            if matched_trials < total_trials * 0.9:
                raise ValueError(
                    f"❌ CRITICAL: Only {matched_trials}/{total_trials} trials have behavioral data!\n"
                    f"   Possible ordinal mismatch between s_{s_id}_{p_id}.asc and expdata_{s_id}_{p_id}.mat"
                )

        # Join trial metadata onto events
        ev_df = ev.frame.sort("onset").join_asof(
            df_trials_combined, left_on='onset', right_on='trial_start', strategy='backward'
        )

        # Assign phase
        ev_df = ev_df.with_columns(
            pl.when((pl.col("onset") >= pl.col("disambig_start")) &
                    (pl.col("onset") < pl.col("disambig_end")))
            .then(pl.lit("disambiguation"))
            .when((pl.col("onset") >= pl.col("mooney_start")) &
                  (pl.col("onset") < pl.col("mooney_end")))
            .then(pl.lit("mooney"))
            .otherwise(pl.lit("inter_stimulus"))
            .alias("phase")
        )

        # Select final columns, ensuring all expected columns exist
        desired_cols = [
            "name", "onset", "offset", "duration", "x","y", "dist_from_center",
            "amplitude", "peak_velocity", "dispersion", "disposition",
            "block_type", "BlockNum", "trial_number", "condition", "phase",
            "ImageName", "DidRespondPas", "NumRepetitionFixationFail", "response_PAS_Q"
        ]
        missing_cols = {col: pl.lit(None) for col in desired_cols if col not in ev_df.columns}
        if missing_cols:
            ev_df = ev_df.with_columns(**missing_cols)

        final_df = ev_df.select(desired_cols)
        ev.frame = final_df

        if debug:
            save_df = _stringify_list_columns(final_df)
            save_path = os.path.join(events_out_dir, csv_name)
            save_df.write_csv(save_path)
            print(f"Saved events with metadata to {save_path}")

    if debug:
        print("\nRunning visual verification...")
        plots.plot_phase_alignment_check(
            events_list=dataset.events,
            fileinfo=dataset.fileinfo,
            raw_data_dir=raw_data_dir,
            save_dir=os.path.join(data_quality_folder, "phase_alignment_checks"),
            labels=trial_labels,
            patterns=asc_patterns,
            colors=phase_palette or PHASE_PALETTE
        )

def _stringify_list_columns(df):
    """Return a copy of df with any List-typed columns converted to strings (for CSV export)."""
    df = df.clone()
    for col, dtype in zip(df.columns, df.dtypes):
        if isinstance(dtype, pl.List):
            df = df.with_columns(
                ("[" + pl.col(col).list.eval(pl.element().cast(pl.String)).list.join(", ") + "]").alias(col)
            )
    return df


def apply_behavioral_filters_and_save(dataset, output_dir,
                                       exclude_subjects, exclude_sessions, exclude_blocks):
    """
    Apply final behavioral exclusion criteria (subject/session/block exclusions,
    PAS response filters, repetition-fixation failures, phase/block-type filters),
    save one cleaned CSV per session, and a combined CSV across all sessions.
    """

    os.makedirs(output_dir, exist_ok=True)
    combined = []

    for i, ev in enumerate(dataset.events):
        s_id = str(dataset.fileinfo['gaze']['session_id'][i]).upper()
        p_id = dataset.fileinfo['gaze']['participant_id'][i]

        if p_id in exclude_subjects:
            continue
        if s_id in exclude_sessions.get(p_id, []):
            continue

        # Keep PAS 0, 2, 3 in both sessions, drop PAS 1
        pas_mask = ~pl.col("response_PAS_Q").is_in([1])

        # Block exclusions specific to this participant/session
        bad_blocks = exclude_blocks.get(p_id, {}).get(s_id, [])
        block_mask = ~pl.col("BlockNum").is_in(bad_blocks)

        final_mask = (
            (pl.col("NumRepetitionFixationFail").fill_null(0) <= 0) &
            (pl.col("DidRespondPas").fill_null(0) != 0) &
            (pl.col("phase") != "inter_stimulus") &
            (pl.col("block_type") != "Practice") &
            pas_mask &
            block_mask
        )
        ev.frame = ev.frame.filter(final_mask)

        prefix = "conscious" if "C" in s_id else "unconscious"
        ev.frame = ev.frame.with_columns(
            pl.when(pl.col("response_PAS_Q") == 0)
              .then(pl.lit(f"{prefix}_unaware"))
              .otherwise(pl.lit(f"{prefix}_aware"))
              .alias("awareness")
        )

        # Save per-session CSV (list columns stringified for CSV compatibility)
        save_df = _stringify_list_columns(ev.frame)
        save_path = os.path.join(output_dir, f"s_{s_id}_{p_id}.csv")
        save_df.write_csv(save_path)

        risky_cols = ["response_PAS_Q", "DidRespondPas", "NumRepetitionFixationFail", "BlockNum"]
        tagged = ev.frame.with_columns(
            session_id=pl.lit(s_id), participant_id=pl.lit(p_id)
        )
        cast_exprs = [pl.col(c).cast(pl.Float64) for c in risky_cols if c in tagged.columns]
        if cast_exprs:
            tagged = tagged.with_columns(cast_exprs)
        combined.append(tagged)

    print(f"Cleaned events saved to {output_dir}")

    if combined:
        combined_df = pl.concat(combined, how="diagonal")
        combined_path = os.path.join(output_dir, "all_events_cleaned.csv")
        _stringify_list_columns(combined_df).write_csv(combined_path)
        print(f"Combined cleaned events saved to {combined_path}")
        return combined_df

    return pl.DataFrame()