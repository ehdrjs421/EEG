# tca_fe.py
import numpy as np

def extract_tca_features(
    spectral_energy_tensor,
    context_window_size,
    step_len_sec,
    window_len_sec
):
    """
    Parameters
    ----------
    spectral_energy_tensor : np.ndarray
        Shape (n_windows, n_channels, n_subbands)
    context_window_size : int
        Number of consecutive windows for TCA
    step_len_sec : float
        Step size in seconds
    window_len_sec : float
        Window length in seconds

    Returns
    -------
    features : np.ndarray
        Shape (n_samples, feature_dim)
    window_end_times : list
        End time (sec) of each feature vector
    """

    ta_features = []
    ca_features = []
    window_end_times = []

    n_windows = spectral_energy_tensor.shape[0]

    for i in range(n_windows - context_window_size + 1):
        pre_tca_block = spectral_energy_tensor[
            i : i + context_window_size, :, :
        ]

        # --- TA ---
        ta_feature = np.mean(pre_tca_block, axis=0)
        ta_features.append(ta_feature.flatten())

        # --- CA ---
        ca_feature = np.mean(pre_tca_block, axis=1)
        ca_features.append(ca_feature.flatten())

        # window end time
        end_time = (i + context_window_size - 1) * step_len_sec + window_len_sec
        window_end_times.append(end_time)

    if not ta_features:
        return None, None

    features = np.hstack([
        np.array(ta_features),
        np.array(ca_features)
    ])

    return features, window_end_times
