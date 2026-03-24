import numpy as np
import pandas as pd


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


def smooth_scores(scores, window_sec=60, step_sec=1):
    """
    SVM decision scores에 이동 평균(Moving Average)을 적용하여 
    순간적인 노이즈로 인한 오탐지를 억제합니다.
    """
    window_len = window_sec // step_sec
    if window_len <= 1:
        return scores

    # pandas의 rolling mean을 사용하여 앞부분 결측치 없이 부드럽게 계산
    return pd.Series(scores).rolling(window=window_len, min_periods=1).mean().values
