import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

from model.poly_svm import PolySVM
from model.oversampling import oversample_seizure
from post_processing.event_extraction import extract_seizure_events
from post_processing.post_filter import apply_post_filter, smooth_scores, adaptive_smooth_scores
from post_processing.seizure_merge import estimate_max_gap_from_one_shot
from post_processing.context_filter import compute_context_threshold
from training_and_adaptation.sampling import stratified_time_sampling


def one_shot_training(
    X, y, df_info,
    random_state=10
):
    random.seed(random_state)

    # ✨ pre-ictal(1) 기준으로 이벤트 추출 (-1 ictal/SPH 구간은 무시)
    seizure_events = extract_seizure_events(y)
    if not seizure_events:
        return None

    chosen_event = random.choice(seizure_events)
    seizure_idx  = list(range(chosen_event[0], chosen_event[1]))

    # ✨ -1 구간 제외: inter-ictal(0)만 비발작으로 사용
    nonseizure_idx = [i for i in df_info[df_info['label'] == 0].index.tolist()
                      if y[i] == 0]

    n_seizure_train    = max(1, min(50, int(len(seizure_idx) * 0.1)))  # ✨ 10->50으로 완화하여 30분 구간 패턴 충분히 학습
    n_nonseizure_train = n_seizure_train * 10  # ✨ 보다 엄격한 오탐지(FA) 학습을 위해 5배->10배 비율로 증가

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
    # 전체 비발작:발작 비율 = 304.9:1 (실측)
    # oversampling(×3) → ≈100:1, train 샘플링(10:1) → ≈10:1
    dynamic_pos_weight = 10.0  # ✨ 20->10 감소: 알람 허들을 높여 오탐지(FA) 억제

    # ✨ train 시 -1 제외 (ictal/SPH 구간 오염 방지)
    valid_train_mask = y_train != -1
    X_train_valid    = X_train_scaled[valid_train_mask]
    y_train_valid    = y_train[valid_train_mask]

    X_train_os, y_train_os = oversample_seizure(
        X_train_valid, y_train_valid, ratio=3
    )

    svm = PolySVM(
        degree=2, coef0=1, C=10.0, gamma=0.5,
        lr=0.001, n_iters=1000,
        loss_weight=True,
        pos_weight=dynamic_pos_weight,
        lr_decay=0.0001
    )
    svm.fit(X_train_os, y_train_os)
    svm.prune_support_vectors(threshold=1e-3)

    # ✨ 예측은 전체 test 시퀀스 그대로 (타임라인 유지)
    raw_scores = svm.decision_function(X_test_scaled)
    decision_scores = adaptive_smooth_scores(raw_scores) # ✨ 적응형 스무딩 (30/60/120s)
    y_pred_raw = (decision_scores > 0.2).astype(int)
    y_pred     = apply_post_filter(y_pred_raw, min_consec=15)  # ✨ 3->15초: 일시적 노이즈 알람 완벽 제거

    # ✨ classification_report는 -1 제외 후 계산
    valid_test_mask = y_test != -1
    report = classification_report(
        y_test[valid_test_mask], y_pred[valid_test_mask],
        output_dict=True, zero_division=0
    )

    # 전체 시퀀스 score (chosen_event 직전 패턴 추출용)
    raw_all_scores = svm.decision_function(scaler.transform(X))
    all_scores     = adaptive_smooth_scores(raw_all_scores) # ✨ 스무딩 반영

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
        context_sec=60,  # ✨ 10->60초: 전조 증상은 수 분에 걸쳐 서서히 나타남
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
        'dynamic_pos_weight'        : dynamic_pos_weight,
    }