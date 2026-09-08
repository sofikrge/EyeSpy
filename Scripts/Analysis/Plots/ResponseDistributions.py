"""How each recording's PAS and pleasantness answers are distributed, one bar per session.

A quick look at the responses themselves, before any eye-tracking: awareness cells are
built from the PAS answers, so a session that hardly ever reports one level is thin in
the analysis no matter how clean its gaze data is. Each bar is 100% of one session's
answers, split by response, so an unusable session is visible while data is still being
collected. Reads the behavioural files only; nothing else in the pipeline has to have run.

Reads:  <behavioural>/expdata_<SESSION>_<PID>.mat   (the Experiment and Extra trials)
Writes: Figures/ResponseDistributions.png
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root, for Settings
from Settings import BEHAVIOURAL_DIR, SECTION_TO_BLOCK, FILTER_PALETTE

# The dataset to plot: Settings.py's own, or another one's behavioural folder.
BEH_DIR = Path("data_rep/my_dataset/behavioural")   # or Path(BEHAVIOURAL_DIR), Settings.py's own
SECTIONS = [s for s in SECTION_TO_BLOCK if s != "Trials_Practice"]  # practice never counts
OUT_PNG = Path("Figures/ResponseDistributions.png")

# One panel per question the dataset actually recorded: the .mat field, its responses in
# scale order, and their labels.
# Pleasantness has no neutral level, hence -2, -1, 1, 2 (the Q/W/E/R keys).
QUESTIONS = [
    ("PAS", "response_PAS_Q", [0, 1, 2, 3],
     ["0  no experience", "1  brief glimpse", "2  almost clear", "3  clear"]),
    ("Pleasantness", "response_PLS_Q", [-2, -1, 1, 2],
     ["-2  very unpleasant", "-1  unpleasant", "1  pleasant", "2  very pleasant"]),
]
NO_ANSWER = "no answer"                  # unanswered, empty or NaN: still part of the 100%
NO_ANSWER_COLOUR = "#c0c0c0"
# Both scales have four levels, so each takes FILTER_PALETTE in scale order (light = low).

# The unaware cell is built from the PAS 0 answers, so a session that reports too few of
# them has no usable cell. PAS 0 is the bottom segment of every stack, which is what makes
# a single dashed line readable: the bar either reaches it or does not.
PAS_0_THRESHOLD = 20                     # % of a session's answers that must be PAS 0
CHECK_SESSIONS = "U"                     # only unconscious sessions supply the unaware cell

# The PAS answers that build each awareness cell downstream. PAS 1 is in neither: the
# pipeline drops it, so it is reported by neither line and only shows up in the bars.
CELLS = [("PAS 0", (0,)), ("PAS 2 or 3", (2, 3))]


def read_field(trial, name):
    """One scalar field, with MATLAB's empty placeholders and NaNs normalised to None."""
    value = getattr(trial, name, None)
    if value is None or (isinstance(value, np.ndarray) and value.size == 0):
        return None
    return None if isinstance(value, float) and np.isnan(value) else value


def read_responses(mat_path):
    """{field: Counter of responses} for one session, over the analysed trial sections.

    A section the session never ran is stored as rows of empty placeholders, so a trial
    without a TrialNum is skipped: counting those would read as thousands of unanswered
    trials rather than as a block that did not happen.
    """
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    expdata = next(v for k, v in mat.items() if k.lower() == "expdata")
    counts = {field: Counter() for _, field, _, _ in QUESTIONS}

    for section in SECTIONS:
        struct = getattr(expdata, section, None)
        if struct is None:
            continue
        for trial in (struct if isinstance(struct, np.ndarray) else [struct]):
            if read_field(trial, "TrialNum") is None:      # ghost trial: empty placeholder
                continue
            for _, field, _, _ in QUESTIONS:
                value = read_field(trial, field)
                counts[field][NO_ANSWER if value is None else int(value)] += 1
    return counts


def plot_panel(ax, title, sessions, counts, levels, labels):
    """One 100% stacked bar per session, in scale order with unanswered trials on top."""
    bottoms = np.zeros(len(sessions))
    totals = np.array([max(sum(c.values()), 1) for c in counts])
    segments = list(zip(levels, labels, FILTER_PALETTE)) + [(NO_ANSWER, NO_ANSWER, NO_ANSWER_COLOUR)]
    for level, label, colour in segments:
        share = np.array([c[level] for c in counts]) / totals * 100
        ax.bar(sessions, share, bottom=bottoms, label=label, color=colour,
               edgecolor="white", linewidth=0.5)
        bottoms += share
    ax.set(title=title, ylabel="% of answers", ylim=(0, 100))
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=8)


def flag_thin_cells(ax, sessions, counts):
    """Mark the sessions reporting PAS 0 on under PAS_0_THRESHOLD% of their trials.

    Returns the list of those sessions; their tick labels are reddened too, so a bar that
    falls short is readable off the figure rather than only off the printed list.
    """
    ax.axhline(PAS_0_THRESHOLD, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(1.005, PAS_0_THRESHOLD / 100, f"{PAS_0_THRESHOLD}%", transform=ax.transAxes,
            va="center", fontsize=7)

    below = [session for session, c in zip(sessions, counts)
             if session[-1] in CHECK_SESSIONS
             and 100 * c[0] / max(sum(c.values()), 1) < PAS_0_THRESHOLD]
    for label in ax.get_xticklabels():
        if label.get_text() in below:
            label.set_color("#b2182b")
    return below


def print_cell_shares(sessions, counts):
    """One line per awareness cell, with the session letters reported apart.

    C and U sessions are the same participants under different viewing conditions, so a
    figure pooling them would describe neither: the unaware cell comes from the U sessions
    and the aware one is mostly C. Each share is pooled over that letter's sessions rather
    than averaged across them, so a session cut short does not count as much as a full one,
    and unanswered trials stay in the denominator, matching the bars above.
    """
    letters = sorted({session[-1] for session in sessions})
    width = max(len(name) for name, _ in CELLS)
    for name, levels in CELLS:
        shares = []
        for letter in letters:
            cells = [c for session, c in zip(sessions, counts) if session[-1] == letter]
            n = sum(c[level] for c in cells for level in levels)
            n_answers = sum(sum(c.values()) for c in cells)
            share = f"{100 * n / max(n_answers, 1):5.1f}%"
            shares.append(f"{letter} = {share}  {f'({n}/{n_answers})':<13}")
        print(f"average {name:<{width}} %   " + "  ".join(shares).rstrip())


files = sorted(BEH_DIR.glob("expdata_*.mat"))
if not files:
    raise SystemExit(f"no expdata_<SESSION>_<PID>.mat files under {BEH_DIR.resolve()}")
sessions = [f"{f.stem.split('_')[2]}{f.stem.split('_')[1]}" for f in files]  # 101C, 101U, ...
responses = [read_responses(f) for f in files]

# A question the dataset never recorded gets no panel: an all-grey one says nothing, and
# the pilot has no pleasantness field at all.
asked = [(title, [r[field] for r in responses], levels, labels)
         for title, field, levels, labels in QUESTIONS
         if any(k != NO_ANSWER for r in responses for k in r[field])]
if not asked:
    raise SystemExit(f"no PAS or pleasantness answers in the .mat files under {BEH_DIR.resolve()}")

fig, axes = plt.subplots(len(asked), 1, squeeze=False,
                         figsize=(max(8, 0.45 * len(files)), 4.5 * len(asked)))
for ax, (title, counts, levels, labels) in zip(axes.flat, asked):
    plot_panel(ax, title, sessions, counts, levels, labels)
    if title == "PAS":                   # only the PAS answers decide the awareness cells
        below = flag_thin_cells(ax, sessions, counts)
        print_cell_shares(sessions, counts)
        print(f"under {PAS_0_THRESHOLD}% PAS 0: {', '.join(below) if below else 'none'}")

fig.suptitle(f"Response distribution per session  -  {BEH_DIR}", fontsize=11)
fig.tight_layout()
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved {OUT_PNG}  ({len(files)} sessions)")
