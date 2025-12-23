import numpy as np
from post_processing.event_extraction import extract_seizure_events

def compute_sample_weights(
    y_true,
    info,
    base_weight=1.0,
    max_weight=10.0
):
    sample_weights = np.ones(len(y_true)) * base_weight

    for (patient, file), group in info.groupby(['patient', 'file']):
        y_file = y_true[group.index]
        events = extract_seizure_events(y_file.values)

        for start, end in events:
            duration = end - start + 1
            for i in range(start, end + 1):
                rel_pos = (i - start) / duration
                weight = base_weight + (max_weight - base_weight) * rel_pos
                sample_weights[group.index[i]] = weight

    return sample_weights
