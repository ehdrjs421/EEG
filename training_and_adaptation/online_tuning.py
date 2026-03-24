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
    max_seizure_samples=30,
    adaptive_min_consec=8,
    max_train_samples=300   # ✨ Forgetting Factor — 최대 train 샘플 수
):
    # ✨ 고확신 pre-ictal(1) 샘플만 재학습에 사용 (-1 제외)
    high_conf_idx = np.where(np.abs(decision_scores) > 0.8)[0]
    seizure_idx   = [i for i in high_conf_idx if y_test[i] == 1]

    if len(seizure_idx) > max_seizure_samples:
        seizure_idx = random.sample(seizure_idx, max_seizure_samples)

    if not seizure_idx:
        return svm, None

    if X_train_scaled is None or y_train is None:
        X_train_scaled = X_test_scaled.reshape(-1, 1)
        y_train        = y_test.copy()

    X_new = X_test_scaled[seizure_idx]
    y_new = y_test[seizure_idx]

    X_aug = np.vstack([X_train_scaled, X_new])
    y_aug = np.concatenate([y_train, y_new])

    # ✨ Forgetting Factor — 최신 max_train_samples개만 유지
    if len(X_aug) > max_train_samples:
        X_aug = X_aug[-max_train_samples:]
        y_aug = y_aug[-max_train_samples:]

    # ✨ -1 제외 후 학습 (ictal/SPH 오염 방지)
    valid_mask   = y_aug != -1
    X_aug_valid  = X_aug[valid_mask]
    y_aug_valid  = y_aug[valid_mask]

    X_aug_os, y_aug_os = oversample_seizure(X_aug_valid, y_aug_valid, ratio=3)

    svm.fit(X_aug_os, y_aug_os)
    svm.prune_support_vectors(threshold=1e-3)

    # ✨ 예측은 전체 시퀀스 그대로 (타임라인 유지)
    scores = svm.decision_function(X_test_scaled)
    y_pred = apply_post_filter(
        (scores > 0.4).astype(int),
        min_consec=15  # ✨ 8->15초: 예측 모델에 맞게 긴 이벤트만 허용
    )

    return svm, y_pred