import numpy as np
from post_processing.event_extraction import extract_seizure_events


CONTEXT_SEC = 10  # 이벤트 직전 몇 초를 볼 것인지


def compute_context_threshold(
    scores,
    tp_event,
    non_seizure_scores,
    context_sec=CONTEXT_SEC,
    step_sec=1
):
    """
    one_shot 단계에서 환자별 초기 context threshold를 계산합니다.

    chosen_event(TP) 직전 score 패턴과
    train non-seizure score 분포를 기반으로
    pre_mean threshold를 설정합니다.

    Parameters
    ----------
    scores : np.ndarray
        전체 시퀀스 score (svm.decision_function(scaler.transform(X)))
    tp_event : tuple (start, end)
        chosen_event (타임스텝 단위)
    non_seizure_scores : np.ndarray
        train non-seizure 샘플들의 score
    context_sec : int
        이벤트 직전 몇 초를 볼 것인지
    step_sec : int
        타임스텝당 초

    Returns
    -------
    dict
        {
            'pre_mean_threshold': float,  ← 이 값 이상이면 FA로 판정
            'slope_threshold'   : float,  ← 이 값 이하면 FA로 판정
            'tp_pre_mean'       : float,  ← TP 직전 score 평균
            'fa_pre_mean'       : float,  ← FA(non-seizure) score 평균
        }
    """
    context_len = context_sec // step_sec

    # TP 패턴 — chosen_event 직전 score
    pre_start  = max(0, tp_event[0] - context_len)
    tp_pre     = scores[pre_start:tp_event[0]]
    tp_pre_mean = float(np.mean(tp_pre)) if len(tp_pre) > 0 else 0.0
    tp_slope    = float(np.polyfit(range(len(tp_pre)), tp_pre, 1)[0]) \
                  if len(tp_pre) >= 3 else 0.0

    # FA 패턴 — train non-seizure score 분포
    fa_pre_mean = float(np.mean(non_seizure_scores)) \
                  if len(non_seizure_scores) > 0 else 0.0
    fa_slope    = 0.0  # non-seizure는 시계열 구조 없음 → slope 추정 불가

    # threshold = TP와 FA 중간값
    # → TP는 통과, FA는 제거
    pre_mean_threshold = (tp_pre_mean + fa_pre_mean) / 2
    slope_threshold    = (tp_slope + (-0.15)) / 2  # -0.15는 데이터 기반 FA 기울기 참고값

    return {
        'pre_mean_threshold': round(pre_mean_threshold, 4),
        'slope_threshold'   : round(slope_threshold, 4),
        'tp_pre_mean'       : round(tp_pre_mean, 4),
        'fa_pre_mean'       : round(fa_pre_mean, 4),
    }


def update_context_threshold(
    scores,
    y_pred,
    current_threshold,
    context_sec=CONTEXT_SEC,
    step_sec=1,
    high_conf=0.8,
    alpha=0.3
):
    """
    online_tuning 이후 고확신 TP/FA 이벤트로 threshold를 갱신합니다.

    Parameters
    ----------
    scores : np.ndarray
        online_tuning 이후 최종 score
    y_pred : np.ndarray
        online_tuning 이후 예측 시퀀스
    current_threshold : dict
        one_shot 단계에서 계산한 초기 threshold
    high_conf : float
        고확신 기준 score (기본값: 0.8)
    alpha : float
        갱신 비율 (0~1, 기본값: 0.3)
        new = alpha * online_value + (1-alpha) * current_value

    Returns
    -------
    dict : 갱신된 threshold
    """
    context_len = context_sec // step_sec
    pred_events = extract_seizure_events(y_pred)

    tp_pre_means, fa_pre_means = [], []
    tp_slopes,    fa_slopes    = [], []

    for p_start, p_end in pred_events:
        # 이벤트 내 score 평균으로 고확신 여부 판단
        event_score_mean = np.mean(scores[p_start:p_end + 1])

        pre_start  = max(0, p_start - context_len)
        pre_scores = scores[pre_start:p_start]

        if len(pre_scores) < 3:
            continue

        slope    = float(np.polyfit(range(len(pre_scores)), pre_scores, 1)[0])
        pre_mean = float(np.mean(pre_scores))

        if event_score_mean >= high_conf:
            # 고확신 예측 → TP로 간주
            tp_pre_means.append(pre_mean)
            tp_slopes.append(slope)
        elif event_score_mean < 0.0:
            # 낮은 score 이벤트 → FA로 간주 (post_filter 통과한 약한 이벤트)
            fa_pre_means.append(pre_mean)
            fa_slopes.append(slope)

    # 갱신할 데이터가 없으면 현재 값 유지
    if not tp_pre_means and not fa_pre_means:
        return current_threshold

    new_tp_pre = float(np.mean(tp_pre_means)) if tp_pre_means else current_threshold['tp_pre_mean']
    new_fa_pre = float(np.mean(fa_pre_means)) if fa_pre_means else current_threshold['fa_pre_mean']

    # Exponential moving average 방식으로 부드럽게 갱신
    updated_tp = alpha * new_tp_pre + (1 - alpha) * current_threshold['tp_pre_mean']
    updated_fa = alpha * new_fa_pre + (1 - alpha) * current_threshold['fa_pre_mean']

    new_pre_mean_threshold = (updated_tp + updated_fa) / 2

    new_slope = float(np.mean(tp_slopes)) if tp_slopes else \
                current_threshold['slope_threshold']
    updated_slope = alpha * ((new_slope + (-0.15)) / 2) + \
                    (1 - alpha) * current_threshold['slope_threshold']

    return {
        'pre_mean_threshold': round(new_pre_mean_threshold, 4),
        'slope_threshold'   : round(updated_slope, 4),
        'tp_pre_mean'       : round(updated_tp, 4),
        'fa_pre_mean'       : round(updated_fa, 4),
    }


def apply_context_filter(
    y_pred,
    scores,
    context_threshold,
    context_sec=CONTEXT_SEC,
    step_sec=1
):
    """
    context threshold를 기반으로 FA 이벤트를 제거합니다.

    FA 판정 조건 (둘 다 만족해야 제거):
      1. 이벤트 직전 score 평균 > pre_mean_threshold
      2. 이벤트 직전 score 기울기 < slope_threshold

    둘 다 만족 → FA로 판정 → 제거
    하나라도 불만족 → 발작으로 유지

    Parameters
    ----------
    y_pred : np.ndarray
    scores : np.ndarray
    context_threshold : dict
    context_sec : int
    step_sec : int

    Returns
    -------
    y_filtered : np.ndarray
    filter_log : list of dict
    """
    context_len = context_sec // step_sec
    pred_events = extract_seizure_events(y_pred)

    y_filtered = y_pred.copy()
    filter_log = []

    pre_mean_thr = context_threshold['pre_mean_threshold']
    slope_thr    = context_threshold['slope_threshold']

    for p_start, p_end in pred_events:
        pre_start  = max(0, p_start - context_len)
        pre_scores = scores[pre_start:p_start]

        if len(pre_scores) < 3:
            continue

        pre_mean = float(np.mean(pre_scores))
        slope    = float(np.polyfit(range(len(pre_scores)), pre_scores, 1)[0])

        # FA 판정: 두 조건 모두 만족해야 제거 (엄격하게)
        is_fa = (pre_mean > pre_mean_thr) and (slope < slope_thr)

        if is_fa:
            y_filtered[p_start:p_end + 1] = 0
            filter_log.append({
                'event_start' : p_start,
                'event_end'   : p_end,
                'pre_mean'    : round(pre_mean, 4),
                'slope'       : round(slope, 4),
                'removed'     : True
            })

    return y_filtered, filter_log