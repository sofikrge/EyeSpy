import polars as pl
from bisect import bisect_left

def parse_blink_intervals(file_path):
    """Extract (onset, offset) tuples for each EBLINK line in an .asc file."""
    intervals = []
    with open(file_path) as f:
        for line in f:
            if line.startswith("EBLINK"):
                parts = line.split()
                intervals.append((int(parts[2]), int(parts[3])))
    return intervals

def parse_trials_from_asc(file_path, labels, patterns):
    """
    Robust trial parser that handles restarts and finds precise phase windows.
    Requires 'labels' and 'patterns' dicts passed from Settings.
    """
    trials_raw = []
    image_map = {}
    phases = []

    pat_trial = patterns['trial']
    pat_image = patterns['image']
    pat_msg   = patterns['msg']

    label_vals = tuple(labels.values())
    with open(file_path, 'r', encoding="utf-8", errors="ignore") as f:
        for line in f:
            if (m := pat_trial.search(line)):
                ts, block, num = int(m.group(1)), m.group(2), int(m.group(3))
                trials_raw.append({"onset": ts, "trial": num, "block_type": block})
            elif (m := pat_image.search(line)):
                ts, img_num = int(m.group(1)), int(m.group(2))
                image_map[ts] = img_num
            elif "MSG" in line:
                if any(c in line for c in label_vals):
                    if (m := pat_msg.search(line)):
                        ts, code = int(m.group(1)), m.group(2)
                        p_type = None
                        if code in (labels['intact'], labels['not_intact']): p_type = "disamb_start"
                        elif code == labels['disamb_end']: p_type = "disamb_end"
                        elif code == labels['mooney_steady']: p_type = "mooney_start"
                        elif code == labels['mooney_end']: p_type = "mooney_end"
                        if p_type:
                            phases.append({"ts": ts, "type": p_type, "code": code})

    img_ts_sorted = sorted(image_map.keys())
    last_ts = max((p['ts'] for p in phases), default=0)
    if img_ts_sorted: last_ts = max(last_ts, img_ts_sorted[-1])
    if trials_raw: last_ts = max(last_ts, trials_raw[-1]['onset'] + 1000)

    final_trials = []
    for i, tr in enumerate(trials_raw):
        tr_next_onset = trials_raw[i+1]['onset'] if i+1 < len(trials_raw) else last_ts
        win_start, win_end = tr['onset'], tr_next_onset

        img_num = None
        if img_ts_sorted:
            idx = bisect_left(img_ts_sorted, win_start)
            if idx < len(img_ts_sorted) and img_ts_sorted[idx] <= win_end:
                img_num = image_map[img_ts_sorted[idx]]

        p_in_win = [p for p in phases if win_start <= p['ts'] <= win_end]
        get_ts = lambda t: next((p['ts'] for p in p_in_win if p['type'] == t), None)
        d_start = get_ts('disamb_start')
        d_end   = get_ts('disamb_end')
        m_start = get_ts('mooney_start')
        m_end   = get_ts('mooney_end')

        cond_code = next((p['code'] for p in p_in_win if p['type'] == 'disamb_start'), None)
        condition = 'intact' if cond_code == labels['intact'] else \
                    'scrambled' if cond_code == labels['not_intact'] else None

        final_trials.append({
            "trial_start": tr['onset'],
            "block_type": tr['block_type'],
            "trial_number": tr['trial'],
            "image_number": img_num,
            "condition": condition,
            "disambig_start": d_start,
            "disambig_end": d_end,
            "mooney_start": m_start,
            "mooney_end": m_end,
            "_has_data": 1 if (d_start and m_start) else 0
        })

    df = pl.DataFrame(final_trials)
    if df.is_empty(): return df

    df = (
        df
        .sort(["block_type", "trial_number", "_has_data", "trial_start"])
        .group_by(["block_type", "trial_number"])
        .tail(1)
        .drop("_has_data")
    )
    return df.sort("trial_start")

