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
from evaluation_and_analysis.visualization import plot_pca_decision_boundary

# 1. 실험 설정
BASE_DATA_PATH = r'C:/Users/ehdrj/Desktop/학교/졸업프로젝트/뇌전증 매트랩/chb-mit-scalp-eeg-database-1.0.0'

RESULT_PATH =  r'C:/Users/ehdrj/Desktop/학교/졸업프로젝트/뇌전증 매트랩/chb-mit-scalp-eeg-database-1.0.0/result'
os.makedirs(RESULT_PATH, exist_ok=True)

RANDOM_SEED = 10
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 2. Feature Extraction
print("🔧 Feature extraction started...")
X, y, df_info = build_dataset(BASE_DATA_PATH,RESULT_PATH)
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
    X_test = one_shot['X_test']
    y_test = one_shot['y_test']
    y_pred_before = one_shot['y_pred']
    decision_scores = one_shot['decision_scores']

# 5. Online Tuning
    svm, y_pred_after = online_tuning(
        svm=svm,
        X_train_scaled=None,   # 내부에서 재구성
        y_train=None,
        X_test_scaled=X_test,
        y_test=y_test,
        decision_scores=decision_scores
    )

    y_pred = y_pred_after if y_pred_after is not None else y_pred_before

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
    results.append({
        'patient': patient_id,
        **metrics,
        'latency': latency,
        'latencies': latencies,
        'vec_sens_60': vec_sens,
        **resource
    })

# 9. 메모리 정리
    del X_pat, y_pat, df_info_pat
    gc.collect()

# 10. 전체 결과 요약
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(RESULT_PATH, "final_results.csv"), index=False)

print("\n✅ Experiment completed.")
print(df_results.mean(numeric_only=True))
