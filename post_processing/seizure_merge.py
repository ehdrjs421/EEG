import numpy as np
from post_processing.event_extraction import extract_seizure_events


def estimate_max_gap_from_one_shot(chosen_event, ratio=0.5, step_sec=1, fallback_sec=30):
    """
    one_shot_train에서 선택된 발작 이벤트 지속시간을 기반으로
    초기 max_gap_sec을 추정합니다. (y_true를 합법적으로 알 수 있는 유일한 시점)

    Parameters
    ----------
    chosen_event : tuple (start, end)
        one_shot_train의 chosen_event (타임스텝 단위)
    ratio : float
        발작 지속시간 대비 max_gap 비율 (기본값: 0.5 → 지속시간의 절반)
    step_sec : int
        타임스텝당 초
    fallback_sec : int
        chosen_event가 없거나 너무 짧을 때 사용할 기본값 (초)

    Returns
    -------
    max_gap_sec : float
    """
    if chosen_event is None:
        return float(fallback_sec)

    duration_sec = (chosen_event[1] - chosen_event[0]) * step_sec
    if duration_sec <= 0:
        return float(fallback_sec)

    # 최솟값 5초 보장 (너무 짧은 발작이 기준이 되는 경우 방지)
    return float(max(duration_sec * ratio, 5.0))


def estimate_max_gap_from_pred(y_pred, ratio=0.5, step_sec=1, fallback_sec=30):
    """
    예측 시퀀스의 발작 이벤트 지속시간 중앙값을 기반으로
    max_gap_sec을 동적으로 계산합니다. (y_true 불필요 → 실제 배포 환경 적용 가능)

    Parameters
    ----------
    y_pred : np.ndarray
        예측 시퀀스 (0 or 1)
    ratio : float
        발작 지속시간 중앙값 대비 max_gap 비율 (기본값: 0.5)
    step_sec : int
        타임스텝당 초
    fallback_sec : float
        예측된 발작이 없을 때 사용할 기본값 (one_shot 기반값 전달 권장)

    Returns
    -------
    max_gap_sec : float
    """
    events = extract_seizure_events(y_pred)
    if not events:
        return float(fallback_sec)

    durations = [(end - start) * step_sec for start, end in events]
    median_duration = float(np.median(durations))

    return float(max(median_duration * ratio, 5.0))


def merge_seizure_events(
    y_pred,
    decision_scores,
    max_gap_sec,
    score_threshold=0.0,
    step_sec=1
):
    """
    예측 시퀀스에서 발작 이벤트를 덩어리화합니다.

    gap 구간이 아래 두 조건을 모두 만족하면 인접한 두 이벤트를 하나로 병합합니다.
      조건 1. gap 길이 ≤ max_gap_sec
      조건 2. gap 구간의 decision_score 평균 ≥ score_threshold
              (score가 음수면 모델이 확신하는 비발작 구간 → 병합 안 함)

    max_gap_sec 결정 흐름 (run_experiment에서 관리):
      1단계 (one_shot): estimate_max_gap_from_one_shot(chosen_event) → 초기값
      2단계 (online 후): estimate_max_gap_from_pred(y_pred, fallback=초기값) → 갱신

    Parameters
    ----------
    y_pred : np.ndarray
        예측 시퀀스 (0 or 1)
    decision_scores : np.ndarray
        SVM decision_function 출력값 (y_pred와 동일 길이)
    max_gap_sec : float
        병합을 허용할 최대 gap 길이 (초 단위)
        estimate_max_gap_from_* 함수로 계산한 값을 전달
    score_threshold : float
        gap 병합을 허용할 최소 decision_score 평균 (기본값: 0.0)
        SVM 결정 경계(0) 기준 — 음수면 비발작 확신, 양수면 발작에 근접
    step_sec : int
        타임스텝당 초 (build_dataset STEP_LEN_SEC=1 기준)

    Returns
    -------
    y_merged : np.ndarray
        덩어리화가 적용된 예측 시퀀스 (0 or 1)
    merged_events : list of (start, end)
        병합된 이벤트 목록 (타임스텝 단위)
    merge_log : list of dict
        병합 이력 (디버깅/분석용)
    """
    events = extract_seizure_events(y_pred)

    if len(events) <= 1:
        return y_pred.copy(), events, []

    merged_events = [events[0]]
    merge_log = []

    for i in range(1, len(events)):
        prev = merged_events[-1]
        curr = events[i]

        gap_start   = prev[1] + 1
        gap_end     = curr[0] - 1
        gap_len_sec = (gap_end - gap_start + 1) * step_sec

        # 조건 1: gap 길이 체크
        if gap_len_sec > max_gap_sec:
            merged_events.append(curr)
            continue

        # 조건 2: gap 구간 score 체크
        gap_scores     = decision_scores[gap_start:gap_end + 1]
        gap_score_mean = float(np.mean(gap_scores)) if len(gap_scores) > 0 else -1.0

        if gap_score_mean >= score_threshold:
            # 병합
            merged_events[-1] = (prev[0], curr[1])
            merge_log.append({
                'merged_from'   : i - 1,
                'merged_to'     : i,
                'gap_start'     : gap_start,
                'gap_end'       : gap_end,
                'gap_len_sec'   : gap_len_sec,
                'gap_score_mean': round(gap_score_mean, 4)
            })
        else:
            merged_events.append(curr)

    # 병합된 이벤트로 y_merged 재구성
    y_merged = np.zeros_like(y_pred)
    for start, end in merged_events:
        y_merged[start:end + 1] = 1

    return y_merged, merged_events, merge_log


def summarize_merge_log(merge_log, used_max_gap):
    """
    병합 이력 요약 통계를 반환합니다. (포트폴리오/분석용)
    """
    if not merge_log:
        return {
            'n_merges'        : 0,
            'mean_gap_len_sec': None,
            'mean_gap_score'  : None,
            'used_max_gap'    : round(used_max_gap, 2)
        }

    gap_lens   = [m['gap_len_sec']     for m in merge_log]
    gap_scores = [m['gap_score_mean']  for m in merge_log]

    return {
        'n_merges'        : len(merge_log),
        'mean_gap_len_sec': round(float(np.mean(gap_lens)), 2),
        'mean_gap_score'  : round(float(np.mean(gap_scores)), 4),
        'used_max_gap'    : round(used_max_gap, 2)   # ✨ 실제 사용된 gap 기준 기록
    }