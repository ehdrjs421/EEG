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
