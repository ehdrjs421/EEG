import numpy as np
from sklearn.metrics import classification_report
from post_processing.event_extraction import extract_seizure_events


def compute_basic_metrics(y_true, y_pred):
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )

    return {
        'accuracy': report['accuracy'],
        'sensitivity': report['1.0']['recall'],
        'specificity': report['0.0']['recall'],
        'f1_seizure': report['1.0']['f1-score']
    }


def evaluate_vector_based_detection(y_true, y_pred, threshold=0.6):
    events = extract_seizure_events(y_true)
    if not events:
        return None

    detected = 0
    for start, end in events:
        duration = end - start + 1
        detected_ratio = np.sum(y_pred[start:end+1]) / duration
        if detected_ratio >= threshold:
            detected += 1

    return detected / len(events)

def compute_false_alarms(y_true, y_pred, recording_hours=None, step_sec=1):
    """
    이벤트 단위 False Alarm을 계산합니다.
 
    False Alarm 정의:
      y_pred 기준 이벤트 중 y_true 발작 구간과 겹치지 않는 이벤트
 
    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    recording_hours : float or None
        전체 녹화 시간(시간 단위) — 제공 시 FA/hour 계산
        None이면 시퀀스 길이로 자동 계산
    step_sec : int
        타임스텝당 초 (기본값: 1)
 
    Returns
    -------
    dict
        {
            'n_fa'         : int,    전체 FA 이벤트 수
            'n_tp_events'  : int,    올바르게 감지한 발작 이벤트 수
            'n_true_events': int,    실제 발작 이벤트 수
            'fa_per_hour'  : float,  시간당 FA 횟수
            'event_sensitivity': float  이벤트 단위 sensitivity
        }
    """
    true_events = extract_seizure_events(y_true)
    pred_events = extract_seizure_events(y_pred)
 
    if recording_hours is None:
        recording_hours = len(y_true) * step_sec / 3600
 
    n_fa = 0
    n_tp = 0
 
    for p_start, p_end in pred_events:
        is_tp = False
        for t_start, t_end in true_events:
            # 겹침 여부 확인
            if p_start <= t_end and p_end >= t_start:
                is_tp = True
                break
        if is_tp:
            n_tp += 1
        else:
            n_fa += 1
 
    fa_per_hour = n_fa / recording_hours if recording_hours > 0 else 0.0
 
    # 이벤트 단위 sensitivity (GT 이벤트 중 감지된 비율)
    detected_true = 0
    for t_start, t_end in true_events:
        for p_start, p_end in pred_events:
            if p_start <= t_end and p_end >= t_start:
                detected_true += 1
                break
    event_sens = detected_true / len(true_events) if true_events else 0.0
 
    return {
        'n_fa'             : n_fa,
        'n_tp_events'      : n_tp,
        'n_true_events'    : len(true_events),
        'n_pred_events'    : len(pred_events),
        'fa_per_hour'      : round(fa_per_hour, 4),
        'event_sensitivity': round(event_sens, 4)
    }