"""Per-eye gaze offset correction.

Runs after pix2deg and before velocity computation. The two eyes sit at a fixed
horizontal offset in this rig, so the tracked eye's position is shifted back by
EYE_OFFSET (see REFERENCES.md).
"""

from Settings import EYE_OFFSET
import polars as pl

def shift_gaze_offset(dataset, eye_offset=EYE_OFFSET):
    """Shift gaze coordinates by a specified offset for left and right eyes."""
    
    for g in dataset.gaze:
        # Tracked eye: pymovements 0.27 doesn't expose it on g.metadata, but the
        # validations frame carries an 'eye' column ('left'/'right') per session.
        te = g.validations.get_column("eye")[0] if g.validations is not None else None

        # Determine eye and offset
        eye = "left" if str(te or "R").strip().upper().startswith("L") else "right" # if empty default to right, normalise naming
        off = eye_offset.get(eye, 0.0) # look up offset for this eye, default to 0 if not found
        
        # Apply offset
        g.samples = g.samples.with_columns(
            position = pl.concat_list([
                pl.col("position").list.get(0) + off,
                pl.col("position").list.get(1)]))
        
    return dataset
