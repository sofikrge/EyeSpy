"""Per-dataset overrides for the replication dataset in data_rep/.

Settings.py applies this file (last, so it wins) whenever EYESPY_DATASET=rep, which
CompleteRun.py sets from its DATASET toggle. The data_rep/ paths, the
DataQualityChecks_rep/ folder and the analysesresults_rep/ results root already follow
from that switch, so this file holds only the parameters that genuinely differ between
the two datasets; everything else is left to Settings.py.

Anything defined in Settings.py can be redefined below by assigning it again, e.g.

    INTERPOLATE_BLINKS = False        # replication uses the blink-drop method
    EXCLUDE_SESSIONS   = {}           # different participants, different exclusions

Reads:  nothing
Writes: nothing
"""

# --- Stage 1 overrides
# Display geometry as amended (06.09.2026): 535.68 x 298.08 mm at 74 cm.
SCREEN = {
    "width_px": 1920, "height_px": 1080,
    "width_cm": 53.568, "height_cm": 29.808,
    "distance_cm": 74.0,
    "origin": "upper left",
    "sampling_rate": 1000}

EYE_OFFSET = {"left": +5.31, "right": -5.31}   # horizontal eye offset, visual degrees
IMAGE_SIZE_DEG = (10.00, 7.51)                 # stimulus extent, visual degrees

# Exclusions, defined here even while empty: without them the replication run would
# inherit the pilot's participant numbers from Settings.py. Fill in as they are decided.
EXCLUDE_SUBJECTS = [103, 109, 120,     # not enough PAS 0 trials, for 109 Talya also turned on light mid block IDK when exactly
]

# Whole sessions, format: { ParticipantID: ['SessionLetter'] }
EXCLUDE_SESSIONS = {
#    201: ['U'],   # reason
}

# Single blocks, format: { ParticipantID: { 'SessionLetter': [BlockNums] } }
# Numbers are the session's running order: 1-4 Experiment, 5-8 Extra (5 = Extra 1).
EXCLUDE_BLOCKS = {106: {'U': [3, 5]}, # moved camera mid block
#                  107: {'U': [7]}, # pupil lost mid block
#                  118: {'U': [3]}, # Pupil lost mid block, so redid C&V
                  123: {'U': [1, 2]}, # told me he saw images outside of the frame so i think he was unfocusing his eyes, told him not to do it afterwards
                  126: {'U': [5]}, # stopped mid block
                  127: {'U': [1, 2, 8]}, # fell asleep in these blocks
                  129: {'U': [2]}, # eyes closing a lot so perhaps too tired
                  

}
#    201: {'C': [1]},   # reason


# --- Stage 2 overrides
# SIGMA, HX/HY and the pymovements dataset are derived at the bottom of Settings.py,
# after this file is applied, so setting the value they come from is enough.
MASK_PPD = 48.22            # pixels per visual degree (SIGMA follows: 24.11 px)
