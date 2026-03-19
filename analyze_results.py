"""
analyze_results.py
------------------
실험 결과 시각화 실행 스크립트
run_experiment.py 실행 후 이 파일을 별도로 실행하세요.
"""
import os
import pandas as pd

from evaluation_and_analysis.visualization import (
    plot_prediction_timeline,
    plot_event_comparison,
    plot_metrics_comparison
)

# ===============================
# 경로 설정
# ===============================
RESULT_PATH = "/content/drive/MyDrive/chb-mit-scalp-eeg-database-1.0.0/result"
VIZ_PATH    = os.path.join(RESULT_PATH, "figures")
os.makedirs(VIZ_PATH, exist_ok=True)

# ===============================
# 1. 환자별 성능 비교 막대그래프 (전체 환자 + 평균)
# ===============================
print("📊 Plotting metrics comparison...")
plot_metrics_comparison(
    result_csv_path=os.path.join(RESULT_PATH, "final_result.csv"),
    save_path=os.path.join(VIZ_PATH, "metrics_comparison.png")
)

# ===============================
# 2. 전체 환자 타임라인 & 이벤트 비교
# ===============================
pred_files = sorted([
    f for f in os.listdir(RESULT_PATH)
    if f.startswith("pred_sequence_") and f.endswith(".csv")
])

for fname in pred_files:
    patient_id = fname.replace("pred_sequence_", "").replace(".csv", "")
    print(f"📈 Plotting {patient_id}...")

    pred_df = pd.read_csv(os.path.join(RESULT_PATH, fname))

    # 타임라인
    plot_prediction_timeline(
        pred_df=pred_df,
        patient_id=patient_id,
        save_path=os.path.join(VIZ_PATH, f"timeline_{patient_id}.png")
    )

    # 이벤트 블록 비교
    plot_event_comparison(
        pred_df=pred_df,
        patient_id=patient_id,
        save_path=os.path.join(VIZ_PATH, f"events_{patient_id}.png")
    )

print(f"\n✅ All figures saved to: {VIZ_PATH}")