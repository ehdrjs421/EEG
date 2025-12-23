import numpy as np

def apply_post_filter(pred, min_consec=3):
    filtered = np.zeros_like(pred)
    i = 0
    while i < len(pred):
        if pred[i] == 1:
            count = 1
            while i + count < len(pred) and pred[i + count] == 1:
                count += 1
            if count >= min_consec:
                filtered[i:i+count] = 1
            i += count
        else:
            i += 1
    return filtered
