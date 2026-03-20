import numpy as np
from post_processing.post_filter import apply_post_filter


def apply_two_stage_filter(
    decision_scores,
    alert_threshold=0.2,
    alert_min_consec=4,
    confirm_threshold=0.4,
    confirm_min_consec=8
):
    """
    두 단계 threshold 필터를 적용합니다.

    1단계 (Alert):
        낮은 threshold + 짧은 min_consec
        → 짧고 약한 발작 (평균 13초) 포착
        → latency 감소 + 미감지 발작 감소

    2단계 (Confirm):
        높은 threshold + 긴 min_consec
        → 강하고 긴 발작 확정
        → FA 방지

    최종 예측 = alert OR confirm

    배경:
        미감지 발작 분석 결과:
        - 평균 지속시간 13초 → min_consec=8에 걸려서 미감지
        - threshold=0.2 기준 최대 연속 6초 → min_consec=4면 포착 가능
        - threshold=0.4 기준 최대 연속 4초 → min_consec=8 미달

    Parameters
    ----------
    decision_scores : np.ndarray
        SVM decision_function 출력값
    alert_threshold : float
        1단계 score 임계값 (기본값: 0.2)
    alert_min_consec : int
        1단계 최소 연속 구간 (기본값: 4초)
    confirm_threshold : float
        2단계 score 임계값 (기본값: 0.4)
    confirm_min_consec : int
        2단계 최소 연속 구간 (기본값: 8초)

    Returns
    -------
    y_pred : np.ndarray
        최종 예측 시퀀스 (0 or 1)
    y_alert : np.ndarray
        1단계 예측 (분석용)
    y_confirm : np.ndarray
        2단계 예측 (분석용)
    """
    # 1단계: alert
    y_alert = apply_post_filter(
        (decision_scores > alert_threshold).astype(int),
        min_consec=alert_min_consec
    )

    # 2단계: confirm
    y_confirm = apply_post_filter(
        (decision_scores > confirm_threshold).astype(int),
        min_consec=confirm_min_consec
    )

    # 최종: alert OR confirm
    y_pred = np.clip(y_alert + y_confirm, 0, 1)

    return y_pred, y_alert, y_confirm