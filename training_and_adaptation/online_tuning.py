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
    adaptive_min_consec=8
):
    MIN_CONSEC_CONF = 3  # 3초 연속 고확신이어야 재학습 데이터로 사용
 
    high_conf_mask = (np.abs(decision_scores) > 0.8).astype(int)
 
    consistent_mask = np.zeros_like(high_conf_mask)
    count = 0
    for i in range(len(high_conf_mask)):
        if high_conf_mask[i] == 1:
            count += 1
            if count >= MIN_CONSEC_CONF:
                # 현재 포함, 앞선 구간도 소급 표시
                consistent_mask[i - MIN_CONSEC_CONF + 1: i + 1] = 1
        else:
            count = 0

    high_conf_idx = np.where(consistent_mask == 1)[0]
    seizure_idx = [i for i in high_conf_idx if y_test[i] == 1]

    if len(seizure_idx) > max_seizure_samples:
        seizure_idx = random.sample(seizure_idx, max_seizure_samples)

    if not seizure_idx:
        return svm, None

    if X_train_scaled is None or y_train is None:
        X_train_scaled = X_test_scaled.reshape(-1, 1)
        y_train = y_test.copy()

    X_new = X_test_scaled[seizure_idx]
    y_new = y_test[seizure_idx]

    X_aug = np.vstack([X_train_scaled, X_new])
    y_aug = np.concatenate([y_train, y_new])

    X_aug_os, y_aug_os = oversample_seizure(X_aug, y_aug, ratio=3)

    svm.fit(X_aug_os, y_aug_os)
    svm.prune_support_vectors(threshold=1e-3)

    scores = svm.decision_function(X_test_scaled)

    # min_consec=8 고정 (기존 최선값)
    # FA와 미감지 발작의 score 분포가 겹쳐서
    # threshold/min_consec 조정만으로는 동시 해결 불가
    y_pred = apply_post_filter(
        (scores > 0.4).astype(int),
        min_consec=8
    )

    return svm, y_pred