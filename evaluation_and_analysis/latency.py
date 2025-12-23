import numpy as np
from post_processing.event_extraction import extract_seizure_events


def compute_latency_in_event(y_true, y_pred, step_sec=1):
    try:
        seizure_idxs = np.where(y_true == 1)[0]
        pred_idxs = np.where(y_pred == 1)[0]

        start = seizure_idxs[0]
        end = seizure_idxs[-1]

        in_event_preds = pred_idxs[
            (pred_idxs >= start) & (pred_idxs <= end)
        ]

        if len(in_event_preds) == 0:
            return None

        return (in_event_preds[0] - start) * step_sec
    except:
        return None


def compute_latency_per_event(y_true, y_pred, step_sec=1):
    events = extract_seizure_events(y_true)
    latencies = []

    for start, end in events:
        pred_idxs = np.where(y_pred == 1)[0]
        in_event = pred_idxs[
            (pred_idxs >= start) & (pred_idxs <= end)
        ]

        if len(in_event) > 0:
            latencies.append((in_event[0] - start) * step_sec)
        else:
            latencies.append(None)

    return latencies
