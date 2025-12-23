import numpy as np
from .bandpass import bandpass_filter

def extract_tca_features(
    eeg_data,
    sfreq,
    sub_bands,
    window_len_samples,
    step_len_samples,
    context_window_size
):
    n_channels = eeg_data.shape[0]
    file_spectral_features = []

    for ch in range(n_channels):
        ch_data = eeg_data[ch]
        ch_features = []
        for low, high in sub_bands:
            filtered = bandpass_filter(ch_data, low, high, sfreq)
            windows = [
                np.sum(np.abs(filtered[i:i+window_len_samples]))
                for i in range(0, len(filtered) - window_len_samples + 1, step_len_samples)
            ]
            ch_features.append(windows)
        file_spectral_features.append(ch_features)

    min_windows = min(len(w) for ch in file_spectral_features for w in ch)
    if min_windows < context_window_size:
        return None, None

    tensor = np.array([
        [sb[:min_windows] for sb in ch]
        for ch in file_spectral_features
    ]).transpose(2, 0, 1)

    ta_features, ca_features, end_times = [], [], []

    for i in range(tensor.shape[0] - context_window_size + 1):
        block = tensor[i:i+context_window_size]

        ta = np.mean(block, axis=0).flatten()
        ca = np.mean(block, axis=1).flatten()

        ta_features.append(ta)
        ca_features.append(ca)

        end_times.append((i + context_window_size - 1))

    return np.hstack([ta_features, ca_features]), end_times
