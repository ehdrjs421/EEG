import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

from model.poly_svm import PolySVM
from model.oversampling import oversample_seizure
from post_processing.event_extraction import extract_seizure_events
from post_processing.post_filter import apply_post_filter
from post_processing.seizure_merge import estimate_max_gap_from_one_shot
from post_processing.context_filter import compute_context_threshold  # ✨
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
    seizure_idx = list(range(chosen_event[0], chosen_event[1]))
    nonseizure_idx = df_info[df_info['label'] == 0].index.tolist()

    n_seizure_train = max(1, min(10, int(len(seizure_idx) * 0.5)))
    n_nonseizure_train = n_seizure_train * 5

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

    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    dynamic_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 15.0
    # 너무 극단적인 값 방지 (5.0 ~ 30.0 범위로 클리핑)
    dynamic_pos_weight = float(np.clip(dynamic_pos_weight, 5.0, 30.0))

    X_train_os, y_train_os = oversample_seizure(
        X_train_scaled, y_train, ratio=3
    )

    svm = PolySVM(
        degree=2, coef0=1, C=10.0, gamma=0.5,
        lr=0.001, n_iters=1000,
        loss_weight=True, pos_weight=15.0
    )
    svm.fit(X_train_os, y_train_os)
    svm.prune_support_vectors(threshold=1e-3)

    decision_scores = svm.decision_function(X_test_scaled)
    y_pred_raw = (decision_scores > 0.2).astype(int)
    y_pred     = apply_post_filter(y_pred_raw, min_consec=3)

    report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )

    # ✨ 전체 시퀀스 score 계산 (chosen_event 직전 패턴 추출용)
    all_scores = svm.decision_function(scaler.transform(X))

    # ✨ one_shot 기반 초기 max_gap
    initial_max_gap = estimate_max_gap_from_one_shot(
        chosen_event=chosen_event,
        ratio=0.5,
        step_sec=1,
        fallback_sec=30
    )

    # ✨ one_shot 기반 초기 context_threshold (patient-specific)
    # chosen_event 직전 패턴(TP) vs train non-seizure score(FA) 비교
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
        'dynamic_pos_weight'        : dynamic_pos_weight,  # ✨
    }