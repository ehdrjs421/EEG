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
    SVM decision scores에 고정 이동 평균(Moving Average)을 적용합니다.
    """
    window_len = window_sec // step_sec
    if window_len <= 1:
        return scores

    # pandas의 rolling mean을 사용하여 앞부분 결측치 없이 부드럽게 계산
    return pd.Series(scores).rolling(window=window_len, min_periods=1).mean().values


def adaptive_smooth_scores(raw_scores, step_sec=1):
    """
    SVM Score의 변동성(Volatility)을 실시간으로 분석하여 
    스무딩 윈도우 크기를 30/60/120초 사이로 동적으로 전환합니다 (Adaptive Smoothing).
    """
    s_raw = pd.Series(raw_scores)
    
    # 1. 지역적 변동성(표준편차) 계산 (최근 5분 기준)
    volatility = s_raw.rolling(window=300, min_periods=1).std().fillna(0).values
    
    # 2. 미리 3가지 버전의 이동 평균 계산 (벡터화 연산으로 고속 처리)
    m30  = s_raw.rolling(window=30, min_periods=1).mean().values
    m60  = s_raw.rolling(window=60, min_periods=1).mean().values
    m120 = s_raw.rolling(window=120, min_periods=1).mean().values
    
    smoothed = np.zeros_like(raw_scores)
    for i in range(len(raw_scores)):
        # 조건 1: 노이즈가 심한(널뛰는) 구간 -> 신중하게 120초 방어
        if volatility[i] > 0.6:
            smoothed[i] = m120[i]
        # 조건 2: 매우 안정적이고 명확한 상승(전조) -> 기민하게 30초 반응
        elif raw_scores[i] > 0.6 and volatility[i] < 0.2:
            smoothed[i] = m30[i]
        # 조건 3: 기본 상황 -> 60초 평균
        else:
            smoothed[i] = m60[i]
            
    return smoothed
