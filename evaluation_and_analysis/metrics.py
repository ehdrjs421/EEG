import numpy as np
from sklearn.metrics import classification_report
from post_processing.event_extraction import extract_seizure_events


def compute_basic_metrics(y_true, y_pred):
    # ✨ -1 구간 제외 후 기본 지표 계산
    valid_mask = y_true != -1
    report = classification_report(
        y_true[valid_mask], y_pred[valid_mask],
        output_dict=True, zero_division=0
    )
    return {
        'accuracy'   : report['accuracy'],
        'sensitivity': report['1.0']['recall'],
        'specificity': report['0.0']['recall'],
        'f1_seizure' : report['1.0']['f1-score']
    }


def evaluate_vector_based_detection(y_true, y_pred, threshold=0.6):
    # pre-ictal(1) 이벤트 기준
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


def evaluate_early_detection(y_true, y_pred, latency_threshold_sec=30, step_sec=1):
    """
    pre-ictal 구간 내에서 N초 이내 감지율.
    """
    events = extract_seizure_events(y_true)
    if not events:
        return None

    detected = 0
    for start, end in events:
        window_end = min(start + latency_threshold_sec // step_sec, end)
        if np.sum(y_pred[start:window_end + 1]) > 0:
            detected += 1

    return round(detected / len(events), 4)


def compute_detection_latency(y_true, y_pred, step_sec=1):
    """
    ✨ 예측 모델용 latency 계산
    - pre-ictal 이벤트 기준으로 첫 감지 시점 계산
    - 감지된 경우만 포함 (미감지 제외)
    - prediction_lead_time 개념:
        양수 = 발작 전 예측 성공 (클수록 좋음)
        음수 = 발작 후 감지 (기존 방식)
    """
    true_events = extract_seizure_events(y_true)
    pred_events = extract_seizure_events(y_pred)

    if not true_events:
        return {
            'median_sec': None, 'mean_sec': None,
            'min_sec': None,    'max_sec': None,
            'n_detected': 0,    'n_total': 0,
            'latencies': []
        }

    latencies = []
    for t_start, t_end in true_events:
        in_event_preds = [
            p_start for p_start, p_end in pred_events
            if p_start <= t_end and p_end >= t_start
        ]
        if in_event_preds:
            first_detection = min(in_event_preds)
            latency = (first_detection - t_start) * step_sec
            latencies.append(latency)

    if not latencies:
        return {
            'median_sec': None, 'mean_sec': None,
            'min_sec': None,    'max_sec': None,
            'n_detected': 0,    'n_total': len(true_events),
            'latencies': []
        }

    return {
        'median_sec' : round(float(np.median(latencies)), 2),
        'mean_sec'   : round(float(np.mean(latencies)), 2),
        'min_sec'    : round(float(np.min(latencies)), 2),
        'max_sec'    : round(float(np.max(latencies)), 2),
        'n_detected' : len(latencies),
        'n_total'    : len(true_events),
        'latencies'  : latencies
    }


def compute_false_alarms(y_true, y_pred, recording_hours=None, step_sec=1):
    """
    ✨ -1 구간(ictal/SPH)에서의 예측은 FA 페널티 면제
    - inter-ictal(0) 구간에서의 오탐만 FA로 카운트
    - pre-ictal(1) 구간과 겹치면 TP
    - ictal/SPH(-1) 구간과 겹치면 무시 (페널티 없음)
    """
    true_events = extract_seizure_events(y_true)
    pred_events = extract_seizure_events(y_pred)

    if recording_hours is None:
        # ✨ -1 구간 제외한 유효 시간으로 recording_hours 계산
        valid_samples = np.sum(y_true != -1)
        recording_hours = valid_samples * step_sec / 3600

    n_fa = 0
    n_tp = 0

    for p_start, p_end in pred_events:
        is_tp     = False
        is_ignore = False

        # TP 체크: pre-ictal(1) 이벤트와 겹치는지
        for t_start, t_end in true_events:
            if p_start <= t_end and p_end >= t_start:
                is_tp = True
                break

        # ✨ FA 면제: -1 구간과 겹치는지
        if not is_tp:
            pred_zone = y_true[p_start:p_end + 1]
            if np.any(pred_zone == -1):
                is_ignore = True

        if is_tp:
            n_tp += 1
        elif not is_ignore:
            n_fa += 1

    fa_per_hour = n_fa / recording_hours if recording_hours > 0 else 0.0

    # event_sensitivity: pre-ictal 이벤트 기준
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