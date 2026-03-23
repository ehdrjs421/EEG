import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

from model.poly_svm import PolySVM
from model.oversampling import oversample_seizure
from post_processing.event_extraction import extract_seizure_events
from post_processing.post_filter import apply_post_filter
from post_processing.seizure_merge import estimate_max_gap_from_one_shot
from post_processing.context_filter import compute_context_threshold
from training_and_adaptation.sampling import stratified_time_sampling


def one_shot_training(
    X, y, df_info,
    random_state=10
):
    random.seed(random_state)

    seizure_events = extract_seizure_events(y)
    if not seizure_events:
        return None

    chosen_event = random.choice(seizure_events)
    seizure_idx  = list(range(chosen_event[0], chosen_event[1]))
    nonseizure_idx = df_info[df_info['label'] == 0].index.tolist()

    n_seizure_train = max(1, min(10, int(len(seizure_idx) * 0.5)))

    # ✨ 비발작 샘플 수를 실제 데이터 비율에 맞게 조정
    # 기존: n_nonseizure = n_seizure * 5 (고정 1:5 비율)
    # 개선: 실제 전체 데이터 비율을 반영하되 최대 샘플 수 제한
    #   실제 비율 = 전체 비발작 / 전체 발작
    #   단, 너무 많으면 학습이 느려지므로 최대 100개로 제한
    total_seizure  = int(np.sum(y == 1))
    total_normal   = int(np.sum(y == 0))
    actual_ratio   = total_normal / total_seizure if total_seizure > 0 else 5.0
    # 비율 그대로 쓰되 최대 100개 제한 (one_shot 경량 학습 철학 유지)
    n_nonseizure_train = min(int(n_seizure_train * actual_ratio), 100)
    n_nonseizure_train = max(n_nonseizure_train, n_seizure_train * 5)  # 최소 5배 보장

    if len(nonseizure_idx) < n_nonseizure_train:
        print("skip")
        return None

    train_idx = (
        random.sample(seizure_idx, n_seizure_train) +
        stratified_time_sampling(
            nonseizure_idx, len(y), n_nonseizure_train
        )
    )

    # Gap 추가 — train 샘플 인접 window를 test에서 제외
    GAP = 2
    exclude = set()
    for idx in train_idx:
        exclude.update(range(idx - GAP, idx + GAP + 1))

    test_idx = sorted(set(range(len(y))) - set(train_idx) - exclude)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ✨ pos_weight 설정 — CHB-MIT 실제 불균형 비율 기반
    # 근거:
    #   전체 데이터 비발작:발작 비율 = 304.9:1 (CHB-MIT 실측값)
    #   oversampling(ratio=3)으로 1차 보정 → 남은 불균형 ≈ 100:1
    #   train 샘플링(5:1)으로 2차 보정   → 남은 불균형 ≈ 20:1
    #   → pos_weight = 20으로 설정
    dynamic_pos_weight = 20.0

    X_train_os, y_train_os = oversample_seizure(
        X_train_scaled, y_train, ratio=3
    )

    svm = PolySVM(
        degree=2, coef0=1, C=10.0, gamma=0.5,
        lr=0.001, n_iters=1000,
        loss_weight=True,
        pos_weight=dynamic_pos_weight,  # ✨ 동적 할당
        lr_decay=0.0001
    )
    svm.fit(X_train_os, y_train_os)
    svm.prune_support_vectors(threshold=1e-3)

    decision_scores = svm.decision_function(X_test_scaled)
    y_pred_raw = (decision_scores > 0.2).astype(int)
    y_pred     = apply_post_filter(y_pred_raw, min_consec=3)

    report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )

    # 전체 시퀀스 score 계산 (chosen_event 직전 패턴 추출용)
    all_scores = svm.decision_function(scaler.transform(X))

    # one_shot 기반 초기 max_gap
    initial_max_gap = estimate_max_gap_from_one_shot(
        chosen_event=chosen_event,
        ratio=0.5,
        step_sec=1,
        fallback_sec=30
    )

    # one_shot 기반 초기 context_threshold (patient-specific)
    non_seizure_train_scores = all_scores[
        [i for i in train_idx if y[i] == 0]
    ]
    initial_context_threshold = compute_context_threshold(
        scores=all_scores,
        tp_event=chosen_event,
        non_seizure_scores=non_seizure_train_scores,
        context_sec=10,
        step_sec=1
    )

    return {
        'svm'                       : svm,
        'scaler'                    : scaler,
        'X_train_scaled'            : X_train_scaled,
        'y_train'                   : y_train,
        'X_test'                    : X_test_scaled,
        'y_test'                    : y_test,
        'y_pred'                    : y_pred,
        'y_pred_raw'                : y_pred_raw,
        'decision_scores'           : decision_scores,
        'report'                    : report,
        'chosen_event'              : chosen_event,
        'initial_max_gap'           : initial_max_gap,
        'initial_context_threshold' : initial_context_threshold,
        'dynamic_pos_weight'        : dynamic_pos_weight,  # ✨ 분석용
    }