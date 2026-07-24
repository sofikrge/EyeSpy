"""Optional blink handling: PCHIP-interpolate gaze position across short blinks.

Used when INTERPOLATE_BLINKS is on in Settings.py, in place of dropping every event
that overlaps a blink. Runs on the raw pixel column before pix2deg, so a blink-spanning
fixation survives as one continuous fixation instead of being split or discarded.

Parameter rationale and sources: REFERENCES.md.
"""

import numpy as np
import polars as pl
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import binary_dilation

def _missing_run_lengths(mask):
    """Given a boolean 'is-missing' array, return an int array where each missing sample
    holds the length of its consecutive missing run (0 where present). Lets us keep short
    blinks and leave long track losses untouched."""
    run_len = np.zeros(mask.shape, dtype=int)
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])   # run boundaries, paired as (start, end)
    for start, end in zip(edges[0::2], edges[1::2]):
        run_len[start:end] = end - start
    return run_len
def _nearest_anchor_index(anchor):
    """For each sample, the index of the nearest anchor (valid, fittable) sample to its
    left and to its right (-1 / len when none on that side). Running max/min over anchor
    indices; used to measure how far apart the two anchors bracketing a fill sample are."""
    n = len(anchor)
    idx = np.arange(n)
    left  = np.maximum.accumulate(np.where(anchor, idx, -1))          # nearest anchor at <= i
    right = np.minimum.accumulate(np.where(anchor, idx, n)[::-1])[::-1]  # nearest anchor at >= i
    return left, right
def _interpolate_segment(t_seg, coord_views, max_gap_samples, margin_samples):
    """PCHIP-fill short blink gaps within ONE contiguous recording segment, in place.

    `coord_views` are numpy views into this segment's coordinate arrays, so writing into
    them updates the parent arrays.
    """
    # Missing = the tracked eye is absent. In a binocular recording the untracked eye is
    # null throughout, so "all coordinates NaN" isolates the tracked eye regardless of
    # which eye it is (keying off coord 0 alone would read a left-eye session, where
    # coord 0 is the right eye and always null, as one giant gap).
    missing = np.stack([np.isnan(c) for c in coord_views]).all(axis=0)
    run_len = _missing_run_lengths(missing)
    short_blink = missing & (run_len <= max_gap_samples)   # real blinks -> fill
    long_gap    = missing & (run_len >  max_gap_samples)   # track loss  -> never fill
    # Widen each short blink by the margin; never fill long-gap samples.
    # (binary_dilation with iterations<1 would fill the whole segment, so guard margin=0.)
    if margin_samples >= 1:
        fill_region = binary_dilation(short_blink, iterations=margin_samples) & ~long_gap
    else:
        fill_region = short_blink & ~long_gap
    # Widest legitimate fill is one capped blink dilated on both sides, so its bracketing
    # anchors sit that far apart. Anchors further apart than this mean a track loss lies
    # between them and the curve would bridge it, so those samples are left as gaps. Side
    # effect: blinks closer together than 2*margin get merged by the dilation and skipped.
    max_bridge = max_gap_samples + 2 * margin_samples + 1
    for v in coord_views:
        anchor = ~np.isnan(v) & ~fill_region     # fit on valid samples outside the fill region
        if anchor.sum() >= 2:                    # PCHIP needs at least two anchors
            curve = PchipInterpolator(t_seg[anchor], v[anchor], extrapolate=False)(t_seg)
            left, right = _nearest_anchor_index(anchor)
            local = (right - left) <= max_bridge          # anchors bracket only a short gap
            fill_here = fill_region & np.isfinite(curve) & local  # isfinite drops PCHIP-unreachable edges
            v[fill_here] = curve[fill_here]
def interpolate_blink_gaps(dataset, max_gap_ms, sampling_rate=1000, margin_ms=200):
    """PCHIP-interpolate gaze position across short blink gaps, in place, before velocity
    and event detection. EyeLink logs no position during a blink, so pymovements loads
    those samples as null. Each blink is widened by `margin_ms` and bridged with a curve
    fitted to the remaining valid samples. See REFERENCES.md for the parameter rationale.

    Three cases are deliberately left alone: blinks longer than `max_gap_ms` (track loss,
    not a blink), gaps at a recording's edge (PCHIP does not extrapolate), and gaps
    spanning two recording segments. Segments are the blocks within a file, separated by
    multi-second timeline gaps, and are interpolated independently so no curve bridges
    one and overwrites real data in the next.
    """
    # At `sampling_rate` Hz each sample spans 1000/rate ms.
    max_gap_samples = int(round(max_gap_ms * sampling_rate / 1000))
    margin_samples = int(round(margin_ms * sampling_rate / 1000))
    sample_period_ms = 1000.0 / sampling_rate
    print(f"\nInterpolating blink gaps up to {max_gap_ms} ms (+/-{margin_ms} ms margin)...")

    for g in dataset.gaze:
        s = g.samples
        t = s["time"].to_numpy()
        # 'pixel' is a list per sample: [x, y] for one eye, or [x, y, ...] if more coords
        # are stored. One editable float array per coordinate (polars null -> NaN; .copy()
        # makes it writable so the per-segment fills below can write in place).
        n_coords = s["pixel"].list.len().max()
        coords = [s.select(pl.col("pixel").list.get(i)).to_series().to_numpy().copy()
                  for i in range(n_coords)]

        # Segment boundaries: a time step larger than one sample period is a recording gap
        # (block boundary), not missing data. Interpolate within each contiguous segment.
        cut = np.flatnonzero(np.diff(t) > sample_period_ms) + 1
        seg_bounds = np.concatenate(([0], cut, [len(t)])).astype(int)
        for a, b in zip(seg_bounds[:-1], seg_bounds[1:]):
            _interpolate_segment(t[a:b], [c[a:b] for c in coords], max_gap_samples, margin_samples)

        # Rebuild the [x, y, ...] list; NaN (long gaps, segment edges, unfilled) -> null.
        s = s.with_columns([pl.Series(f"_c{i}", coords[i]) for i in range(n_coords)])
        g.samples = s.with_columns(
            pl.concat_list([pl.col(f"_c{i}").fill_nan(None) for i in range(n_coords)]).alias("pixel")
        ).drop([f"_c{i}" for i in range(n_coords)])

    return dataset
