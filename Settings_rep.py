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
EYE_OFFSET = {"left": +5.31, "right": -5.31}   # horizontal eye offset, visual degrees

# Exclusions, defined here even while empty: without them the replication run would
# inherit the pilot's participant numbers from Settings.py. Fill in as they are decided.
EXCLUDE_SUBJECTS = [        # whole participants, e.g. 104, 106
]

# Whole sessions, format: { ParticipantID: ['SessionLetter'] }
EXCLUDE_SESSIONS = {
#    201: ['U'],   # reason
}

# Single blocks, format: { ParticipantID: { 'SessionLetter': [BlockNums] } }
EXCLUDE_BLOCKS = {106: {'U': [3, 5]}, # moved camera mid block
                  107: {'U': [7]}, # pupil lost mid block
                  109: {'U': [2]}, # Talya accidentally turned on light and then turned off again, IDK when exactly
                  118: {'U': [3]}, # Pupil lost mid block, so redid C&V
                  123: {'U': [1, 2]}, # told me he saw images outside of the frame so i think he was unfocusing his eyes, told him not to do it afterwards
                  

}
#    201: {'C': [1]},   # reason


# --- Stage 2 overrides
# SIGMA is half the pixels-per-degree, and Settings.py has already derived it from its
# own MASK_PPD by the time this file is applied, so both have to be set here.
MASK_PPD = 48.22            # pixels per visual degree
SIGMA    = MASK_PPD / 2.0   # 24.11 px, the fixation-map Gaussian blur radius
