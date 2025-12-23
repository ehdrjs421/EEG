import random
import numpy as np
from model.oversampling import oversample_seizure
from post_processing.post_filter import apply_post_filter


def online_tuning(
    svm,
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test,
    decision_scores,
    max_seizure_samples=30
):
    high_conf_idx = np.where(np.abs(decision_scores) > 0.8)[0]
    seizure_idx = [i for i in high_conf_idx if y_test[i] == 1]

    if len(seizure_idx) > max_seizure_samples:
        seizure_idx = random.sample(seizure_idx, max_seizure_samples)

    if not seizure_idx:
        return svm, None

    X_new = X_test_scaled[seizure_idx]
    y_new = y_test[seizure_idx]

    X_aug = np.vstack([X_train_scaled, X_new])
    y_aug = np.concatenate([y_train, y_new])

    X_aug_os, y_aug_os = oversample_seizure(X_aug, y_aug, ratio=3)

    svm.fit(X_aug_os, y_aug_os)
    svm.prune_support_vectors(threshold=1e-3)

    scores = svm.decision_function(X_test_scaled)
    y_pred = apply_post_filter(
        (scores > 0.4).astype(int),
        min_consec=4
    )

    return svm, y_pred
