import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from training_and_adaptation.one_shot_train import one_shot_training
from post_processing.post_filter import adaptive_smooth_scores

def calculate_permutation_importance(patient_id, data_dir, n_repeats=5):
    """
    특정 환자 데이터를 사용하여 피처 그룹별 기여도를 분석합니다.
    """
    # 1. 데이터 로드
    X_path = os.path.join(data_dir, f"X_{patient_id}.npy")
    y_path = os.path.join(data_dir, f"y_{patient_id}.npy")
    info_path = os.path.join(data_dir, f"df_{patient_id}.pkl")

    if not os.path.exists(X_path):
        print(f"❌ {patient_id} 데이터를 찾을 수 없습니다.")
        return None

    X = np.load(X_path)
    y = np.load(y_path)
    df_info = pd.read_pickle(info_path)

    # 2. Base 모델 학습 및 평가 (Adaptive Smoothing 반영)
    print(f"🚀 {patient_id} 모델 학습 중...")
    one_shot = one_shot_training(X, y, df_info, patient_id)
    if one_shot is None:
        return None

    svm = one_shot['svm']
    X_test_scaled = one_shot['X_test_scaled']
    y_test = one_shot['y_test']
    
    # -1 (Ignore/SPH) 제외 마스크
    valid_mask = y_test != -1
    y_test_valid = y_test[valid_mask]

    # Baseline 성능 측정
    raw_scores = svm.decision_function(X_test_scaled)
    # ✨ 우리가 고도해한 adaptive smoothing 적용
    decision_scores = adaptive_smooth_scores(raw_scores)
    y_pred = (decision_scores > 0.2).astype(int)
    baseline_f1 = f1_score(y_test_valid, y_pred[valid_mask], zero_division=0)
    print(f"✅ Baseline F1-Score: {baseline_f1:.4f}")

    # 3. 피처 그룹 정의 (TCA-FE + SLOPE + VAR + CA)
    # TA(0~111), SLOPE(112~223), VAR(224~335), CA(336~356)
    feature_groups = {
        'TA (Band Power)': list(range(0, 112)),
        'SLOPE (Trend)':  list(range(112, 224)),
        'VAR (Volatility)': list(range(224, 336)),
        'CA (Cross-Channel)': list(range(336, 357))
    }

    importances = {}

    # 4. Permutation Importance 계산
    for group_name, indices in feature_groups.items():
        print(f"🔍 '{group_name}' 기여도 분석 중...")
        drop_scores = []
        
        for _ in range(n_repeats):
            # 특정 피처 그룹만 무작위로 섞음
            X_shuffled = X_test_scaled.copy()
            shuffled_indices = np.random.permutation(len(X_shuffled))
            X_shuffled[:, indices] = X_shuffled[shuffled_indices][:, indices]

            # 섞인 시퀀스로 예측 수행
            raw_scores_shuf = svm.decision_function(X_shuffled)
            scores_shuf = adaptive_smooth_scores(raw_scores_shuf)
            y_pred_shuf = (scores_shuf > 0.2).astype(int)
            
            f1_shuf = f1_score(y_test_valid, y_pred_shuf[valid_mask], zero_division=0)
            drop_scores.append(baseline_f1 - f1_shuf)

        importances[group_name] = np.mean(drop_scores)

    return importances, baseline_f1

def plot_importance(importances, patient_id):
    """
    기여도 분석 결과를 시각화합니다.
    """
    df = pd.DataFrame(list(importances.items()), columns=['Feature Group', 'Importance (F1 Drop)'])
    df = df.sort_values(by='Importance (F1 Drop)', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance (F1 Drop)', y='Feature Group', data=df, palette='viridis')
    plt.title(f"Feature Contribution Analysis (Permutation Importance) - {patient_id}")
    plt.xlabel("Importance (F1-score Drop when Shuffled)")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = f"feature_importance_{patient_id}.png"
    plt.savefig(save_path)
    print(f"📊 분석 결과 이미지가 '{save_path}'에 저장되었습니다.")
    plt.show()

if __name__ == "__main__":
    # 데이터 경로 설정 (코랩 환경에 맞춰 수정 가능)
    DATA_DIR = "./data/processed"
    if not os.path.exists(DATA_DIR):
        DATA_DIR = "."  # 로컬 테스트용
        
    TARGET_PATIENT = "chb01"
    
    results = calculate_permutation_importance(TARGET_PATIENT, DATA_DIR)
    if results:
        plot_importance(results[0], TARGET_PATIENT)
