# References

Methods papers behind specific choices in the pipeline. Each entry names the
parameter or step it supports.

## Blink handling

Dankner, Y., Shalev, L., Carrasco, M., & Yuval-Greenberg, S. (2017). Prestimulus
inhibition of saccades in adults with and without attention-deficit/hyperactivity
disorder as an index of temporal expectations. *Psychological Science, 28*(7),
835–850. https://doi.org/10.1177/0956797617694863

> The PCHIP + peri-blink-margin technique behind `BLINK_MARGIN_MS` (`Settings.py`).

Wass, S. V., Smith, T. J., & Johnson, M. H. (2013). Parsing eye-tracking data of
variable quality to provide accurate fixation duration estimates in infants and
adults. *Behavior Research Methods, 45*(1), 229–241.

> Interpolates gaze *position* across gaps up to 150 ms — the upper bound behind
> `MAX_BLINK_INTERP_MS`. (Tobii's I-VT gap fill-in defaults to 75 ms and must stay
> shorter than a blink; 150 ms is the most permissive value we found support for.)

Kret, M. E., & Sjak-Shie, E. E. (2019). Preprocessing pupil size data: Guidelines
and code. *Behavior Research Methods, 51*(3), 1336–1342.

> The same interpolation recipe applied to pupil rather than position data.

## Analysis

Guo et al. (2024) — double group-mean centring in cross-classified multilevel
models. *(Full citation to be filled in.)*

> The centring used in `Scripts/Analysis/Checks/CentreGazeTypicality.py`:
> `x − x̄_participant − x̄_image + x̄_grand`.

## Software

pymovements — https://pymovements.readthedocs.io

> Drives Stage 1 (loading, `pix2deg`, `pos2vel`, IVT event detection). Version
> 0.27.0.
