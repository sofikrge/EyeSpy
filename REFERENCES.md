# Decisions and sources

Why the parameters in `Settings.py` are what they are. The code holds the values and a
one-line pointer; the reasoning lives here.

Entries marked *(no source recorded)* are choices whose justification has not been
written down yet.

The study is registered: `preregistration/Preregistration.pdf` (registered 10.08.2026)
and `preregistration/Amendment to Preregistration.pdf` (06.09.2026, display-geometry
rounding only). Where the registration fixes a value it is the source, and the value is
not a free parameter — changing it is a deviation and has to be reported as one.

---

## Blink handling

`INTERPOLATE_BLINKS` picks between two mutually exclusive methods.

**Interpolate (`True`)** is the primary, **preregistered** method, and the current
setting: PCHIP-interpolate gaze position across short blink gaps before event detection,
so a blink-spanning fixation survives as one fixation. The registration fixes all three
parameters:

> "Blinks will be identified using the EyeLink blink-detection algorithm and extended by
> 200 ms buffers on each side. The padded blink intervals will be reconstructed with a
> shape-preserving piecewise cubic Hermite interpolating polynomial (Dankner et al.,
> 2017). Only blinks up to 150 ms will be interpolated while longer blinks will be
> treated as missing values."

**Filter (`False`)** is the robustness alternative: detect events first, then drop any
fixation or saccade overlapping a blink. It is *not* the registered method — running it
as the primary analysis is a deviation.

The two subsections below are why the registered values are what they are; they were
derived from this dataset rather than borrowed.

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
| `FIX_VELOCITY_THRESHOLD` | 30 deg/s | Registered I-VT threshold (Salvucci & Goldberg, 2000; in line with Lublinsky et al., 2025) |
| `MIN_FIX_DURATION_MS` | 50 ms | Registered minimum fixation duration, same source |
| `VALIDATION_ACCURACY_AVG_THRESHOLD` | 1.0 deg | Registered block-level calibration criterion |
| `VALIDATION_ACCURACY_MAX_THRESHOLD` | 1.5 deg | Registered block-level calibration criterion |
| `CENTER_RADIUS_DG` | 1.5 deg | Registered central exclusion radius, to control centre bias (Lublinsky et al., 2025) |
| `EYE_OFFSET` | +/-5.44 deg | Pilot rig. The registered value is +/-5.31 deg (amended from 5.32) and lives in `Settings_rep.py` |
| `BUFFER_FIX` / `BUFFER_SAC` | 51 / 60 ms | Blink buffer for the filter method only; not registered |

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

This is the pilot rig. The registered value is `MASK_PPD = 48.22` (amended from 48.25),
giving `SIGMA = 24.11` px (amended from 24.125) — the 0.5 deg circular buffer the
registration specifies around each fixation. Both live in `Settings_rep.py`.

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

## Statistical model and multiple comparisons

Registered in full, so none of this is a free choice.

### Mixed model

> "Participants and images will be included as random intercepts to capture individual
> and stimulus-specific variability. Additionally, the slopes for the disambiguator type
> will be added as random effects. Fixed effects will include awareness (aware and
> unaware) and disambiguator type (intact and scrambled). Degrees of freedom will be
> estimated using the Satterthwaite approximation."

Fitted in jamovi (GAMLj), bobyqa, Satterthwaite df, Wald CIs:

```
NSS ~ 1 + Awareness + ReferenceMap + Awareness:ReferenceMap
      + (1 + ReferenceMap | Participant) + (1 + ReferenceMap | Image)
```

The random slope is on **ReferenceMap** (= disambiguator type), not on Awareness. The DV
is the trial-level NSS from `NSS_<mode>/NSS_CrossPhase_LongFormat.csv`.

Known ambiguity: the registered sentence does not name the grouping factor for the slope.
Slopes on both grouping factors and a slope on Participant only are both non-singular and
agree on the interaction, but they move the unaware simple effect (p = .057 vs .035).
Slopes on both is the reading taken here — it matches the immediately preceding clause,
which names participants and images together, and it has the better AIC (15381.7 vs
15397.5). Report the other as a sensitivity analysis.

### Tree-BH correction

Bogomolov, Peterson, Benjamini & Sabatti (2021), *Hypotheses on a tree: New error rates
and testing strategies*, Biometrika 108(3), 575-590; Zeevi, Catzman, Benjamini & Mudrik
(2025), *Correction for multiple comparisons should be ubiquitous*, Nature Human
Behaviour 9(12), 2407-2408. The tree is registered as Figure 6:

- **Level 1** (family of 2): disambiguator type x awareness interaction; disambiguator
  type main effect. BH at q = 0.05.
- **Level 2** (family of 2): simple effect of disambiguator type within aware, and within
  unaware. These are children of the **interaction node only** — the main-effect node has
  no children — so level 2 is tested only if the interaction is selected, at the
  selection-adjusted level `q2 = q * R1 / m1`, where `R1` is the number of level-1
  hypotheses rejected and `m1 = 2`.

Each family gets **one** threshold, applied to BH-adjusted p-values: q = 0.05 at level 1,
and q2 at level 2. Within a family BH is a step-up procedure, so the i-th smallest raw p
is compared to `q * i / m`; adjusting the p-values folds that ranking in so every member
of the family is compared to the same number. Because it is step-up, the two members are
coupled: if the larger adjusted p clears the threshold, both are rejected.

With the current results (disambiguator main effect not rejected, so R1 = 1 and
q2 = 0.025):

| Level | Test | raw p | BH-adj | Threshold | |
|---|---|---|---|---|---|
| 1 | Interaction | .00047 | .00094 | 0.05 | passes |
| 1 | Disambiguator main effect | .118 | .118 | 0.05 | fails |
| 2 | Simple effect within aware | .0073 | .0145 | 0.025 | passes |
| 2 | Simple effect within unaware | .0569 | .0569 | 0.025 | fails |

The unaware simple effect is the larger of its family, so its adjusted p is its raw p: it
needs raw p <= 0.025 to survive. It does not reach that under either slope reading
(.057 with slopes on both, .035 with the participant slope only).

Computed by hand from the procedure above; the `TreeBH` package could not be installed
here, so re-check against it before reporting.

## Software

pymovements 0.27.0, https://pymovements.readthedocs.io. Drives Stage 1: loading,
`pix2deg`, `pos2vel`, IVT event detection, microsaccades, event properties.
