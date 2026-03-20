import numpy as np
from scipy.signal import butter, sosfilt


# ✨ 필터 계수 캐시 — 동일 파라미터 반복 호출 시 재계산 방지
_sos_cache = {}


def get_sos_filter(lowcut, highcut, fs, order=5):
    """
    SOS 필터 계수를 계산하고 캐시에 저장합니다.
    동일한 (lowcut, highcut, fs, order) 조합은 재계산하지 않아요.
    """
    key = (lowcut, highcut, fs, order)
    if key not in _sos_cache:
        nyq = 0.5 * fs
        low  = lowcut / nyq
        high = highcut / nyq

        if low <= 0:
            sos = butter(order, high, btype='low', output='sos')
        elif high >= 1:
            sos = butter(order, low, btype='high', output='sos')
        else:
            sos = butter(order, [low, high], btype='band', output='sos')

        _sos_cache[key] = sos

    return _sos_cache[key]


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    SOS 기반 bandpass 필터 (기존 lfilter 대비 수치 안정성 향상 + 속도 개선)

    Parameters
    ----------
    data : np.ndarray (1D)
    lowcut, highcut : float
    fs : float
    order : int

    Returns
    -------
    filtered : np.ndarray (1D)
    """
    sos = get_sos_filter(lowcut, highcut, fs, order)
    return sosfilt(sos, data)