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
from evaluation_and_analysis.metrics import compute_basic_metrics, evaluate_vector_based_detection
from evaluation_and_analysis.latency import compute_latency_in_event, compute_latency_per_event
from evaluation_and_analysis.resource_analysis import analyze_model_resources
from evaluation_and_analysis.visualization import plot_polysvm_decision_boundary

from post_processing.seizure_merge import (
    estimate_max_gap_from_one_shot,
    estimate_max_gap_from_pred,
    merge_seizure_events,
    summarize_merge_log
)
# 1. 실험 설정

BASE_DATA_PATH ="/content/drive/MyDrive/chb-mit-scalp-eeg-database-1.0.0"
RESULT_PATH ="/content/drive/MyDrive/chb-mit-scalp-eeg-database-1.0.0/result"

# BASE_DATA_PATH = r'C:/Users/ehdrj/Desktop/학교/졸업프로젝트/뇌전증 매트랩/chb-mit-scalp-eeg-database-1.0.0'
# RESULT_PATH =  r'C:/Users/ehdrj/Desktop/학교/졸업프로젝트/뇌전증 매트랩/chb-mit-scalp-eeg-database-1.0.0/result'

os.makedirs(RESULT_PATH, exist_ok=True)

RANDOM_SEED = 10
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

MERGE_GAP_RATIO       = 0.5   # 발작 지속시간 대비 max_gap 비율
MERGE_SCORE_THRESHOLD = 0.0   # gap 구간 score 평균 기준 (SVM 결정 경계)
STEP_SEC              = 1     # 타임스텝 = 1초 (build_dataset STEP_LEN_SEC 기준)
FALLBACK_GAP_SEC      = 30    # chosen_event가 없을 때 사용할 기본값
 
# 2. Feature Extraction
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
results_before = []

# 3. Patient Loop
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


    svm = one_shot['svm']
    scaler = one_shot['scaler']
    X_train_scaled = one_shot['X_train_scaled']
    y_train = one_shot['y_train']
    X_test = one_shot['X_test']
    y_test = one_shot['y_test']
    y_pred_before = one_shot['y_pred']
    decision_scores = one_shot['decision_scores']
    chosen_event    = one_shot['chosen_event']

    init_max_gap = estimate_max_gap_from_one_shot(
        chosen_event,
        ratio=MERGE_GAP_RATIO,
        step_sec=STEP_SEC,
        fallback_sec=FALLBACK_GAP_SEC
    )
    print(f"  📏 Init max_gap: {init_max_gap:.1f}s "
          f"(chosen_event duration: {(chosen_event[1]-chosen_event[0])*STEP_SEC}s)")
 
# 5. Online Tuning
    svm, y_pred_after = online_tuning(
        svm=svm,
        X_train_scaled=X_train_scaled,   # 내부에서 재구성
        y_train=y_train,
        X_test_scaled=X_test,
        y_test=y_test,
        decision_scores=decision_scores
    )

    y_pred = y_pred_after if y_pred_after is not None else y_pred_before

    final_scores     = svm.decision_function(X_test)
    adaptive_max_gap = estimate_max_gap_from_pred(
        y_pred,
        ratio=MERGE_GAP_RATIO,
        step_sec=STEP_SEC,
        fallback_sec=init_max_gap
    )
    print(f"  📏 Adaptive max_gap: {adaptive_max_gap:.1f}s "
          f"(updated from y_pred median duration)")
 
    # ✨ 6. Seizure Merge (덩어리화)
    y_pred_merged, merged_events, merge_log = merge_seizure_events(
        y_pred=y_pred,
        decision_scores=final_scores,
        max_gap_sec=adaptive_max_gap,
        score_threshold=MERGE_SCORE_THRESHOLD,
        step_sec=STEP_SEC
    )
    merge_summary = summarize_merge_log(merge_log, used_max_gap=adaptive_max_gap)
 
    print(f"  🔗 Merge | "
          f"events: {len(merged_events)} | "
          f"n_merges: {merge_summary['n_merges']} | "
          f"mean_gap: {merge_summary['mean_gap_len_sec']}s")
 
    pred_df = pd.DataFrame({
        "time_idx": np.arange(len(y_test)),
        "y_true": y_test,
        "y_pred_before": y_pred_before,
        "y_pred_after": y_pred_after if y_pred_after is not None else y_pred_before,
        "decision_score": decision_scores,
        "patient": patient_id
    })

    pred_path = os.path.join(
    RESULT_PATH,
    f"pred_sequence_{patient_id}.csv"
    )
    pred_df.to_csv(pred_path, index=False)
    
# 6. Evaluation
    metrics = compute_basic_metrics(y_test, y_pred)
    vec_sens = evaluate_vector_based_detection(y_test, y_pred, threshold=0.9)
    latency = compute_latency_in_event(y_test, y_pred)
    latencies = compute_latency_per_event(y_test, y_pred)

# 7. Resource Analysis
    resource = analyze_model_resources(
        model=svm,
        X_test=X_test,
        save_path=os.path.join(RESULT_PATH, f"svm_{patient_id}.joblib")
    )

# 8. 결과 저장
    # results.append({
    #     'patient': patient_id,
    #     **metrics,
    #     'latency': latency,
    #     'latencies': latencies,
    #     'vec_sens_60': vec_sens,
    #     **resource
    # })
    results.append({
        'patient'          : patient_id,
        # 기존 메트릭 (merge 전)
        **{f"{k}_before": v for k, v in metrics_before.items()},
        'latency_before'   : latency_before,
        'vec_sens_before'  : vec_sens_before,
        # ✨ 덩어리화 후 메트릭
        **{f"{k}_merged": v for k, v in metrics_merged.items()},
        'latency_merged'   : latency_merged,
        'vec_sens_merged'  : vec_sens_merged,
        # ✨ 병합 통계 (환자별 adaptive gap 추적)
        'init_max_gap'     : round(init_max_gap, 2),
        'adaptive_max_gap' : round(adaptive_max_gap, 2),
        'n_merges'         : merge_summary['n_merges'],
        'mean_gap_len_sec' : merge_summary['mean_gap_len_sec'],
        'mean_gap_score'   : merge_summary['mean_gap_score'],
        **resource
    })

# 9. 메모리 정리
    del X_pat, y_pat, df_info_pat
    gc.collect()

# 10. 전체 결과 요약
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
