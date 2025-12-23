import pandas as pd
from sklearn.metrics import classification_report
from .post_filter import apply_post_filter

def tune_post_filter(y_true, y_pred_raw, max_window=10, metric='f1'):
    scores = []
    best_score = -1
    best_k = 1
    best_pred = None

    for k in range(1, max_window + 1):
        y_filtered = apply_post_filter(y_pred_raw, min_consec=k)
        report = classification_report(
            y_true, y_filtered,
            output_dict=True, zero_division=0
        )

        f1 = report['1.0']['f1-score']
        scores.append({
            'k': k,
            'sensitivity': report['1.0']['recall'],
            'specificity': report['0.0']['recall'],
            'f1': f1
        })

        if report['1.0'][f'{metric}-score'] > best_score:
            best_score = report['1.0'][f'{metric}-score']
            best_k = k
            best_pred = y_filtered

    return best_k, best_score, best_pred, pd.DataFrame(scores)
