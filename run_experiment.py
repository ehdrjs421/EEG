import os
import gc
import random
import numpy as np
import pandas as pd


# ===============================
# Feature Extraction
# ===============================
from feature_extraction.build_dataset import build_dataset

# ===============================
# Training & Adaptation
# ===============================
from training_and_adaptation.sequential_loader import load_patient_data
from training_and_adaptation.one_shot_train import one_shot_training
from training_and_adaptation.online_tuning import online_tuning

# ===============================
# Evaluation & Analysis
# ===============================
from evaluation_and_analysis.metrics import compute_basic_metrics, evaluate_vector_based_detection, compute_false_alarms, evaluate_early_detection
from evaluation_and_analysis.latency import compute_latency_in_event, compute_latency_per_event
from evaluation_and_analysis.resource_analysis import analyze_model_resources
from evaluation_and_analysis.visualization import plot_polysvm_decision_boundary

# ✨ Seizure Merge (덩어리화)
from post_processing.seizure_merge import (
    estimate_max_gap_from_one_shot,
    estimate_max_gap_from_pred,
    merge_seizure_events,
    summarize_merge_log
)

# ✨ Context Filter (patient-specific FA 제거)
from post_processing.context_filter import (
    update_context_threshold,
    apply_context_filter
)

# ===============================
# 1. 실험 설정
# ===============================

BASE_DATA_PATH = "/content/drive/MyDrive/chb-mit-scalp-eeg-database-1.0.0"
RESULT_PATH    = "/content/drive/MyDrive/chb-mit-scalp-eeg-database-1.0.0/result"

os.makedirs(RESULT_PATH, exist_ok=True)

RANDOM_SEED = 10
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ✨ 덩어리화 공통 파라미터
MERGE_GAP_RATIO       = 0.5   # 발작 지속시간 대비 max_gap 비율
MERGE_SCORE_THRESHOLD = 0.0   # gap 구간 score 평균 기준 (SVM 결정 경계)
STEP_SEC              = 1     # 타임스텝 = 1초 (build_dataset STEP_LEN_SEC 기준)
FALLBACK_GAP_SEC      = 30    # chosen_event가 없을 때 사용할 기본값

# ===============================
# 2. Feature Extraction
# ===============================
# 이미 전처리된 파일이 하나라도 있으면 build_dataset 스킵
existing_files = [
    f for f in os.listdir(RESULT_PATH)
    if f.startswith("X_chb") and f.endswith(".npy")
]

if existing_files:
    print("✅ 전처리된 데이터 발견 — Feature extraction 스킵")
    patient_ids = sorted([
        f.replace("X_", "").replace(".npy", "") for f in existing_files
    ])
else:
    print("🔧 Feature extraction started...")
    X, y, df_info = build_dataset(BASE_DATA_PATH, RESULT_PATH)
    print("✅ Feature extraction completed.")
    patient_ids = sorted(df_info['patient'].unique())

results = []

# ===============================
# 3. Patient Loop
# ===============================
for patient_id in patient_ids:
    print(f"\n📂 Processing patient: {patient_id}")

    X_pat, y_pat, df_info_pat = load_patient_data(RESULT_PATH, patient_id)

    if X_pat is None:
        continue

    # 4. One-shot Training
    one_shot = one_shot_training(X_pat, y_pat, df_info_pat)
    if one_shot is None:
        print("⚠️ One-shot training skipped.")
        continue

    svm             = one_shot['svm']
    scaler          = one_shot['scaler']
    X_train_scaled  = one_shot['X_train_scaled']
    y_train         = one_shot['y_train']
    X_test          = one_shot['X_test']
    y_test          = one_shot['y_test']
    y_pred_before   = one_shot['y_pred']
    decision_scores = one_shot['decision_scores']
    chosen_event    = one_shot['chosen_event']
    init_max_gap    = one_shot['initial_max_gap']
    dynamic_pos_weight = one_shot['dynamic_pos_weight']  # ✨
    context_threshold  = one_shot['initial_context_threshold']

    print(f"  chosen_event 지속시간: {(chosen_event[1]-chosen_event[0])*STEP_SEC}초 "
          f"→ init_max_gap={init_max_gap:.1f}s | "
          f"pos_weight={dynamic_pos_weight:.1f} | "
          f"ctx_threshold={context_threshold['pre_mean_threshold']:.3f}")

    # 5. Online Tuning
    svm, y_pred_after = online_tuning(
        svm=svm,
        X_train_scaled=X_train_scaled,
        y_train=y_train,
        X_test_scaled=X_test,
        y_test=y_test,
        decision_scores=decision_scores
    )

    y_pred = y_pred_after if y_pred_after is not None else y_pred_before

    # ✨ 2단계: online tuning 이후 max_gap 갱신
    final_scores     = svm.decision_function(X_test)
    adaptive_max_gap = estimate_max_gap_from_pred(
        y_pred,
        ratio=MERGE_GAP_RATIO,
        step_sec=STEP_SEC,
        fallback_sec=init_max_gap
    )

    # ✨ 2단계: online tuning 이후 context_threshold 갱신
    context_threshold = update_context_threshold(
        scores=final_scores,
        y_pred=y_pred,
        current_threshold=context_threshold,
        context_sec=10,
        step_sec=STEP_SEC,
        high_conf=0.8,
        alpha=0.3
    )
    print(f"  ctx_threshold 갱신 → pre_mean={context_threshold['pre_mean_threshold']:.3f} | "
          f"slope={context_threshold['slope_threshold']:.3f}")

    # ✨ 6. Seizure Merge (덩어리화)
    y_pred_merged, merged_events, merge_log = merge_seizure_events(
        y_pred=y_pred,
        decision_scores=final_scores,
        max_gap_sec=adaptive_max_gap,
        score_threshold=MERGE_SCORE_THRESHOLD,
        step_sec=STEP_SEC
    )
    merge_summary = summarize_merge_log(merge_log, used_max_gap=adaptive_max_gap)

    # ✨ 7. Context Filter (patient-specific FA 제거)
    y_pred_filtered, filter_log = apply_context_filter(
        y_pred=y_pred_merged,
        scores=final_scores,
        context_threshold=context_threshold,
        context_sec=10,
        step_sec=STEP_SEC
    )
    n_filtered = sum(1 for f in filter_log if f['removed'])
    print(f"  Context Filter | 제거된 이벤트: {n_filtered}개")

    print(f"  🔗 Merge | "
          f"events: {len(merged_events)} | "
          f"n_merges: {merge_summary['n_merges']} | "
          f"mean_gap: {merge_summary['mean_gap_len_sec']}s")

    # 예측 시퀀스 저장
    pred_df = pd.DataFrame({
        "time_idx"        : np.arange(len(y_test)),
        "y_true"          : y_test,
        "y_pred_before"   : y_pred_before,
        "y_pred_after"    : y_pred_after if y_pred_after is not None else y_pred_before,
        "y_pred_merged"   : y_pred_merged,
        "y_pred_filtered" : y_pred_filtered,   # ✨ context filter 결과
        "decision_score"  : final_scores,
        "patient"         : patient_id
    })

    pred_path = os.path.join(RESULT_PATH, f"pred_sequence_{patient_id}.csv")
    pred_df.to_csv(pred_path, index=False)

    # 8. Evaluation — before / merged / filtered 비교
    metrics_before   = compute_basic_metrics(y_test, y_pred)
    metrics_merged   = compute_basic_metrics(y_test, y_pred_merged)
    metrics_filtered = compute_basic_metrics(y_test, y_pred_filtered)  # ✨

    vec_sens_before   = evaluate_vector_based_detection(y_test, y_pred,          threshold=0.9)
    vec_sens_merged   = evaluate_vector_based_detection(y_test, y_pred_merged,   threshold=0.9)
    vec_sens_filtered = evaluate_vector_based_detection(y_test, y_pred_filtered, threshold=0.9)  # ✨

    latency_before   = compute_latency_in_event(y_test, y_pred)
    latency_merged   = compute_latency_in_event(y_test, y_pred_merged)
    latency_filtered = compute_latency_in_event(y_test, y_pred_filtered)  # ✨

    latencies_before   = compute_latency_per_event(y_test, y_pred)
    latencies_merged   = compute_latency_per_event(y_test, y_pred_merged)
    latencies_filtered = compute_latency_per_event(y_test, y_pred_filtered)  # ✨

    # ✨ False Alarm 계산 (before / merged / filtered)
    fa_before   = compute_false_alarms(y_test, y_pred,          step_sec=STEP_SEC)
    fa_merged   = compute_false_alarms(y_test, y_pred_merged,   step_sec=STEP_SEC)
    fa_filtered = compute_false_alarms(y_test, y_pred_filtered, step_sec=STEP_SEC)  # ✨

    print(f"  FA | before: {fa_before['fa_per_hour']:.2f}/h "
          f"-> merged: {fa_merged['fa_per_hour']:.2f}/h "
          f"-> filtered: {fa_filtered['fa_per_hour']:.2f}/h")

    # ✨ Early Detection (before / merged / filtered)
    early30_before   = evaluate_early_detection(y_test, y_pred,          latency_threshold_sec=30, step_sec=STEP_SEC)
    early30_merged   = evaluate_early_detection(y_test, y_pred_merged,   latency_threshold_sec=30, step_sec=STEP_SEC)
    early30_filtered = evaluate_early_detection(y_test, y_pred_filtered, latency_threshold_sec=30, step_sec=STEP_SEC)  # ✨
    early60_before   = evaluate_early_detection(y_test, y_pred,          latency_threshold_sec=60, step_sec=STEP_SEC)
    early60_merged   = evaluate_early_detection(y_test, y_pred_merged,   latency_threshold_sec=60, step_sec=STEP_SEC)
    early60_filtered = evaluate_early_detection(y_test, y_pred_filtered, latency_threshold_sec=60, step_sec=STEP_SEC)  # ✨

    print(f"  Early Det | 30s: {early30_before:.3f}->{early30_merged:.3f}->{early30_filtered:.3f} | "
          f"60s: {early60_before:.3f}->{early60_merged:.3f}->{early60_filtered:.3f}")

    # 9. Resource Analysis
    resource = analyze_model_resources(
        model=svm,
        X_test=X_test,
        save_path=os.path.join(RESULT_PATH, f"svm_{patient_id}.joblib")
    )

    # 10. 결과 저장
    results.append({
        'patient'              : patient_id,
        # before (online tuning 후)
        'sensitivity_before'   : metrics_before['sensitivity'],
        'specificity_before'   : metrics_before['specificity'],
        'f1_seizure_before'    : metrics_before['f1_seizure'],
        'latency_before'       : latency_before,
        'vec_sens_before'      : vec_sens_before,
        'fa_per_hour_before'   : fa_before['fa_per_hour'],
        'event_sens_before'    : fa_before['event_sensitivity'],
        'early30_before'       : early30_before,
        # merged (덩어리화 후)
        'sensitivity_merged'   : metrics_merged['sensitivity'],
        'specificity_merged'   : metrics_merged['specificity'],
        'f1_seizure_merged'    : metrics_merged['f1_seizure'],
        'latency_merged'       : latency_merged,
        'vec_sens_merged'      : vec_sens_merged,
        'fa_per_hour_merged'   : fa_merged['fa_per_hour'],
        'event_sens_merged'    : fa_merged['event_sensitivity'],
        'early30_merged'       : early30_merged,
        # filtered (context filter 후)
        'sensitivity_filtered' : metrics_filtered['sensitivity'],
        'specificity_filtered' : metrics_filtered['specificity'],
        'f1_seizure_filtered'  : metrics_filtered['f1_seizure'],
        'latency_filtered'     : latency_filtered,
        'vec_sens_filtered'    : vec_sens_filtered,
        'fa_per_hour_filtered' : fa_filtered['fa_per_hour'],
        'event_sens_filtered'  : fa_filtered['event_sensitivity'],
        'early30_filtered'     : early30_filtered,
        # 공통
        'n_true_events'        : fa_before['n_true_events'],
        # 병합 통계
        'init_max_gap'         : round(init_max_gap, 2),
        'adaptive_max_gap'     : round(adaptive_max_gap, 2),
        'n_merges'             : merge_summary['n_merges'],
        # context threshold
        'ctx_pre_mean_thr'     : context_threshold['pre_mean_threshold'],
        # 모델 리소스
        'model_kb'             : resource.get('model_kb'),
        'pred_time_s'          : resource.get('pred_time_s'),
        'testset_time_s'       : resource.get('testset_time_s'),
    })

    # 10. 메모리 정리
    del X_pat, y_pat, df_info_pat
    gc.collect()

# ===============================
# 11. 전체 결과 요약
# ===============================
csv_path = os.path.join(RESULT_PATH, "final_result.csv")
df_new = pd.DataFrame(results)

if not df_new.empty:
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_results = pd.concat([df_old, df_new], ignore_index=True)
        df_results = df_results.drop_duplicates(subset=['patient'], keep='last')
    else:
        df_results = df_new

    df_results['patient_num'] = df_results['patient'].str.extract(r'(\d+)').astype(int)
    df_results = df_results.sort_values('patient_num').drop(columns='patient_num')
    df_results.to_csv(csv_path, index=False)

print("\n✅ Experiment completed.")
print(df_results.mean(numeric_only=True))