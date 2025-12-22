import mne
import numpy as np

# =========================
# Configuration
# =========================
TARGET_SFREQ = 128  # Hz, 논문 기반

CHANNELS_TO_USE = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'FT10-T8', 'P8-O2'
]

MIN_REQUIRED_CHANNELS = 8


# =========================
# Preprocessing Functions
# =========================

def load_raw_edf(edf_path):
    """
    EDF 파일 로드
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # 채널 이름 정리 (네 코드 그대로)
    raw.rename_channels(lambda s: s.strip().replace('.', ''))
    return raw


def select_channels(raw):
    """
    CHANNELS_TO_USE 중 실제 존재하는 채널만 선택
    """
    available_channels = [ch for ch in CHANNELS_TO_USE if ch in raw.ch_names]

    if len(available_channels) < MIN_REQUIRED_CHANNELS:
        raise ValueError("Insufficient common EEG channels")

    raw.pick(available_channels)
    return raw, len(available_channels)


def resample_eeg(raw):
    """
    EEG 리샘플링
    """
    raw.resample(TARGET_SFREQ, verbose=False)
    eeg = raw.get_data()  # (n_channels, n_samples)
    sfreq = raw.info['sfreq']
    return eeg, sfreq


def preprocess_edf(edf_path):
    """
    EDF → 전처리된 EEG 반환

    Returns
    -------
    eeg : np.ndarray
        shape = (n_channels, n_samples)
    sfreq : float
    n_channels : int
    """
    raw = load_raw_edf(edf_path)
    raw, n_channels = select_channels(raw)
    eeg, sfreq = resample_eeg(raw)

    return eeg, sfreq, n_channels
