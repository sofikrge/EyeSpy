# Decisions and sources

Why the parameters in `Settings.py` are what they are. The code holds the values and a
one-line pointer; the reasoning lives here.

Entries marked *(no source recorded)* are choices whose justification has not been
written down yet. Fill them in before the preregistration.

---

## Blink handling

`INTERPOLATE_BLINKS` picks between two mutually exclusive methods.

**Filter (`False`)** is the primary, preregistered method: detect events first, then
drop any fixation or saccade overlapping a blink.

**Interpolate (`True`)** is a data-justified robustness alternative, not a replication
of any one study: PCHIP-interpolate gaze position across short blink gaps before event
detection, so a blink-spanning fixation survives as one fixation. The PCHIP plus
peri-blink-margin technique is standard practice; the parameter values below come from
this dataset.

### `BLINK_MARGIN_MS = 200`

How far each blink is widened before interpolating, because peri-blink samples are
unreliable and must not anchor the curve.

Validated on this dataset's own contamination profile rather than borrowed: pupil size
and gaze velocity depart roughly 60 ms before blink onset and take about 150 ms to
recover afterwards, so a symmetric margin has to cover ~150 ms. 200 ms clears it; the
old 51 ms filter buffer does not. Run
`Scripts/Preprocessing/Diagnostics/BlinkContaminationProfile.py` to regenerate the
blink-locked averages behind this.

The cost is small: at 200 ms only about 3% of the Mooney phase (the DV) is
reconstructed. See `BlinkInterpolationFraction.py`.

### `MAX_BLINK_INTERP_MS = 150`

Blinks longer than this are treated as track loss and left as gaps.

150 ms is the most permissive value we found support for on gaze **position** rather
than pupil: Tobii's I-VT gap fill-in defaults to 75 ms and must stay "shorter than a
blink", and Wass, Smith & Johnson (2013) interpolate position up to 150 ms.

It sits well above this dataset's median blink (~89 ms across the 24 analysed sessions),
so it rescues about 77% of genuine short blinks. Raising it to 500 ms would add only
~1,500 longer blinks, and those are full eyelid closures where the eye can move behind
the lid: exactly what position interpolation must not fabricate when the DV is a gaze
location. `BlinkDurationDistribution.py` shows the cap sitting in the trough between the
physiological-blink mode and the track-loss tail.

### Sources

Dankner, Y., Shalev, L., Carrasco, M., & Yuval-Greenberg, S. (2017). Prestimulus
inhibition of saccades in adults with and without attention-deficit/hyperactivity
disorder as an index of temporal expectations. *Psychological Science, 28*(7), 835-850.
https://doi.org/10.1177/0956797617694863

Wass, S. V., Smith, T. J., & Johnson, M. H. (2013). Parsing eye-tracking data of
variable quality to provide accurate fixation duration estimates in infants and adults.
*Behavior Research Methods, 45*(1), 229-241.

Kret, M. E., & Sjak-Shie, E. E. (2019). Preprocessing pupil size data: Guidelines and
code. *Behavior Research Methods, 51*(3), 1336-1342.

---

## Event detection and filtering

| Parameter | Value | Why |
|---|---|---|
| `FIX_VELOCITY_THRESHOLD` | 30 deg/s | IVT threshold *(no source recorded)* |
| `MIN_FIX_DURATION_MS` | 50 ms | *(no source recorded)* |
| `VALIDATION_ACCURACY_AVG_THRESHOLD` | 1.0 deg | Sessions above this are dropped *(no source recorded)* |
| `VALIDATION_ACCURACY_MAX_THRESHOLD` | 1.5 deg | *(no source recorded)* |
| `CENTER_RADIUS_DG` | 1.5 deg | Central exclusion radius; inherited from Shaked's pipeline |
| `EYE_OFFSET` | +/-5.44 deg | Per-eye screen offset in this rig |
| `BUFFER_FIX` / `BUFFER_SAC` | 51 / 60 ms | Blink buffer used by the filter method |

---

## NSS analysis

### `DISPERSION_DDOF = 0`

Population SD, not sample SD, for z-normalisation. One of several deliberate
MATLAB-parity choices, kept so results stay comparable with the original
implementation. The others are round-half-away-from-zero rounding, 1-based pixel
indexing, and `gaussian_filter(mode="reflect", truncate=2.0)` to match `imgaussfilt`
with symmetric padding.

### `MASK_PPD = 48.55`, canvas 800x600

Pixels per visual degree and the fixation-map canvas. Inherited from the original
MATLAB implementation so the maps are directly comparable.

### Subject thresholds

| Parameter | Value | Why |
|---|---|---|
| `MIN_SUBJ_PER_IMAGE_NSS` | 2 | LOSO scoring needs at least one other subject |
| `MIN_SUBJ_PER_IMAGE_CROSS` | 2 | Same, for the cross-phase reference maps |
| `MIN_IMAGES_PER_CELL_CROSS` | 15 | Drops a participant's awareness x reference cell when too thin |

### `NAN_POLICY_CROSS = "permissive"`

Ignore a missing reference map and average whatever is present, rather than returning
NaN for the whole image (`"matlab_strict"`).

### Centring the typicality covariate

`CentreGazeTypicality.py` applies double group-mean centring,
`x - x_participant - x_image + x_grand`. The cross-phase data is cross-classified, so
centring on one grouping alone would leave the other's variance in the covariate.

Guo et al. (2024), on double group-mean centring in cross-classified multilevel models.
*(Full citation to be filled in.)*

## Software

pymovements 0.27.0, https://pymovements.readthedocs.io. Drives Stage 1: loading,
`pix2deg`, `pos2vel`, IVT event detection, microsaccades, event properties.
