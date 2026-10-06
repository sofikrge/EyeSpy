"""Per-dataset values for the pilot dataset in Data.nosync/pilot/.

Settings.py applies this file (last, so it wins) whenever EYESPY_DATASET=pilot. The
Data.nosync/pilot/ paths and the pilot/ output subfolders already follow from that
choice, so this file holds only the parameters that genuinely differ between the two
datasets (the same names as Settings_rep.py); everything else is left to Settings.py.

Reads:  nothing
Writes: nothing
"""

# --- Stage 1
# Screen geometry of the pilot rig, in the units pymovements wants (cm).
SCREEN = {
    "width_px": 1920, "height_px": 1080,
    "width_cm": 53.2, "height_cm": 29.8,
    "distance_cm": 74.0,
    "origin": "upper left",
    "sampling_rate": 1000}

EYE_OFFSET = {"left": +5.44, "right": -5.44}   # horizontal eye offset, visual degrees
IMAGE_SIZE_DEG = (9.99, 7.50)                  # stimulus extent, visual degrees

EXCLUDE_SUBJECTS = [
# 104, 106, 109, 110, 112, 118, 120, # left dominant eye
]

# Whole sessions, format: { ParticipantID: ['SessionLetter'] }
EXCLUDE_SESSIONS = {
#    104: ['U'], # unfocused eyes sometimes
#    105: ['U'], # unfocused eyes sometimes
#    106: ['U'], # unfocused eyes sometimes
    107: ['U'], # low PAS 0 trials
    111: ['U'], # low PAS 0 trials
    110: ['U'], # low PAS 0 trials
    118: ['U'], # low PAS 0 trials
#    112: ['C'], # unfocused eyes sometimes
}

# Single blocks, format: { ParticipantID: { 'SessionLetter': [BlockNums] } }
# Block numbers are the session's RUNNING ORDER, the way they are written down during
# recording: 1-4 are the Experiment blocks, 5 onwards are the Extra blocks (5 = Extra 1).
# The saved BlockNum column restarts at 1 in the Extra block, so the numbers here are not
# that column; TrialMetadata.apply_behavioral_filters_and_save translates them, and warns
# about any number the session does not actually have.
EXCLUDE_BLOCKS = {
#    112: {'U': [4, 5]},             # Unfocused eyes sometimes
    117: {'C': [1]},                # Technical mistake
    119: {'U': [1, 2, 3, 4, 5, 6]}  # Unfocused eyes sometimes
}


# --- Stage 2
# SIGMA, HX/HY and the pymovements dataset are derived at the bottom of Settings.py,
# after this file is applied, so setting the value they come from is enough.
MASK_PPD = 48.55            # pixels per visual degree (SIGMA follows: 24.275 px)
