import numpy as np
from scipy.signal import butter, lfilter

# =========================
# Configuration
# =========================

SUB_BANDS = [
    (0.5, 4),
    (4, 8),
    (8, 12),
    (12, 16),
    (16, 20),
    (20, 24),
    (24, 28)
]

FILTER_ORDER = 5


# =========================
# Filtering Functions
# =========================

def bandpass_filter(signal, lowcut, highcut, fs, order=FILTER_ORDER):
    """
    Apply bandpass / lowpass / highpass filter
    (네 원본 코드 그대로)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if low <= 0:
        b, a = butter(order, high, btype='low')
    elif high >= 1:
        b, a = butter(order, low, btype='high')
    else:
        b, a = butter(order, [low, high], btype='band')

    return lfilter(b, a, signal)


def apply_subband_filter(eeg, fs):
    """
    EEG → Sub-band filtered EEG

    Parameters
    ----------
    eeg : np.ndarray
        shape = (n_channels, n_samples)

    Returns
    -------
    subband_eeg : np.ndarray
        shape = (n_channels, n_subbands, n_samples)
    """
    n_channels, n_samples = eeg.shape
    n_subbands = len(SUB_BANDS)

    subband_eeg = np.zeros((n_channels, n_subbands, n_samples))

    for ch in range(n_channels):
        for sb, (low, high) in enumerate(SUB_BANDS):
            subband_eeg[ch, sb, :] = bandpass_filter(
                eeg[ch, :], low, high, fs
            )

    return subband_eeg
