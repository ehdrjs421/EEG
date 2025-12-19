import numpy as np

# =========================
# Configuration
# =========================

WINDOW_LEN_SEC = 2      # 2초
STEP_LEN_SEC = 1        # 1초
TARGET_SFREQ = 128      # Hz (논문 기반)

WINDOW_LEN_SAMPLES = WINDOW_LEN_SEC * TARGET_SFREQ
STEP_LEN_SAMPLES = STEP_LEN_SEC * TARGET_SFREQ


# =========================
# Windowing Functions
# =========================

def compute_spectral_energy(subband_eeg):
    """
    Compute spectral energy using sliding window

    Parameters
    ----------
    subband_eeg : np.ndarray
        shape = (n_channels, n_subbands, n_samples)

    Returns
    -------
    energy_tensor : np.ndarray
        shape = (n_windows, n_channels, n_subbands)
    """
    n_channels, n_subbands, n_samples = subband_eeg.shape

    n_windows = (n_samples - WINDOW_LEN_SAMPLES) // STEP_LEN_SAMPLES + 1
    energy_tensor = []

    for w in range(n_windows):
        start = w * STEP_LEN_SAMPLES
        end = start + WINDOW_LEN_SAMPLES

        window_energy = np.zeros((n_channels, n_subbands))

        for ch in range(n_channels):
            for sb in range(n_subbands):
                window = subband_eeg[ch, sb, start:end]
                window_energy[ch, sb] = np.sum(np.abs(window))

        energy_tensor.append(window_energy)

    return np.array(energy_tensor)
