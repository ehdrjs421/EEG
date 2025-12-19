import os
import gc
import random
import time
import joblib

# 결과 저장 리스트
one_shot_results_before = []
one_shot_results = []

# 데이터 저장 경로
base_path = '/content/drive/MyDrive/data'
# Ensure the directory exists
os.makedirs(base_path, exist_ok=True)

# It's assumed that X, y, df_info are already created from the feature extraction step
# We'll simulate saving them to match the original structure, or you can skip if already saved.
# For demonstration, we'll save X, y, df_info for each patient first.
patient_ids = df_info['patient'].unique()
for p_id in patient_ids:
    patient_df_info = df_info[df_info['patient'] == p_id]
    patient_indices = patient_df_info.index.tolist()
    patient_X = X[patient_indices]
    patient_y = y[patient_indices]

    np.save(os.path.join(base_path, f"X_{p_id}.npy"), patient_X)
    np.save(os.path.join(base_path, f"y_{p_id}.npy"), patient_y)
    patient_df_info.to_pickle(os.path.join(base_path, f"df_{p_id}.pkl"))


patient_ids = sorted([f.split('_')[1].split('.')[0] for f in os.listdir(base_path) if f.startswith('X_')])
random.seed(10)

# Variables to store data for the last patient for PCA visualization
last_X_train_scaled = None
last_y_train = None
last_X_test_scaled = None
last_y_test = None
last_svm_model = None

for patient_id in patient_ids:
    print(f"\n📂 [환자 처리 중] {patient_id}")

    try:
        # ✅ 데이터 불러오기
        X_pat = np.load(os.path.join(base_path, f"X_{patient_id}.npy")).astype(np.float32)
        y_pat = np.load(os.path.join(base_path, f"y_{patient_id}.npy"))
        df_info_pat = pd.read_pickle(os.path.join(base_path, f"df_{patient_id}.pkl")).reset_index(drop=True)
    except:
        print(f"❌ [불러오기 실패] {patient_id}")
        continue

    seizure_events = extract_seizure_events(y_pat)

    # Handle cases where no seizure events are found
    if not seizure_events:
        print(f"⚠️ [발작 이벤트 없음] {patient_id}. 스킵합니다.")
        continue

    chosen_event = random.choice(seizure_events)

    seizure_idx = list(range(chosen_event[0],chosen_event[1]))
    nonseizure_idx = list(df_info_pat[df_info_pat['label'] == 0].index)

    if len(seizure_idx) < 1 or len(nonseizure_idx) < 5:
        print(f"⚠️ [데이터 부족] {patient_id}")
        continue
    # 비례 학습 샘플 수 설정
    n_seizure_train = max(1, min(10, int(len(seizure_idx) * 0.5)))
    n_nonseizure_train = n_seizure_train * 5

    if len(nonseizure_idx) < n_nonseizure_train:
        print(f"⚠️ 비발작 샘플 부족 → 스킵")
        continue

    train_idx = random.sample(seizure_idx, n_seizure_train) + stratified_time_sampling(nonseizure_idx,len(y_pat), n_nonseizure_train)
    test_idx = sorted(set(range(len(y_pat))) - set(train_idx))

    X_train, y_train = X_pat[train_idx], y_pat[train_idx]
    X_test, y_test = X_pat[test_idx], y_pat[test_idx]

    # 정규화
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # 오버샘플링 (정의 필요)
    X_train_os, y_train_os = oversample_seizure(X_train_scaled, y_train, ratio=3)

    # 초기 학습
    svm_model = PolySVM(degree=2, coef0=1, C=10.0, gamma=0.5, lr=0.001, n_iters=1000,
                        loss_weight=True, pos_weight=15.0)
    svm_model.fit(X_train_os, y_train_os)
    svm_model.prune_support_vectors(threshold=1e-3)

    # 예측
    decision_scores = svm_model.decision_function(X_test_scaled)
    y_pred_raw = (decision_scores > 0.2).astype(int)


    # 후처리
    y_pred = y_pred_raw.copy()

    for i in range(len(y_pred_raw)):
        if y_pred_raw[i] == 1:
            count = 1
            for j in range(1, 3):  # post_filter_window = 3
                if i + j < len(y_pred_raw) and y_pred_raw[i + j] == 1:
                    count += 1
                else:
                    break
            if count < 3:
                y_pred[i] = 0

    # ✅ 튜닝 전 결과 저장
    y_pred_before = y_pred.copy()

       # 온라인 튜닝
    high_conf_idx = np.where(np.abs(decision_scores) > 0.8)[0]
    seizure_tuning_idx = [i for i in high_conf_idx if y_test[i] == 1]
    max_seizure_tuning = 30
    if len(seizure_tuning_idx) > max_seizure_tuning:
        seizure_tuning_idx = random.sample(seizure_tuning_idx, max_seizure_tuning)

    print(f"🔄 온라인 튜닝 샘플 수: {len(seizure_tuning_idx)}")
    if len(seizure_tuning_idx) > 0:
        X_new = X_test_scaled[seizure_tuning_idx]
        y_new = y_test[seizure_tuning_idx]
        X_aug = np.vstack([X_train_scaled, X_new])
        y_aug = np.concatenate([y_train, y_new])
        X_aug_os, y_aug_os = oversample_seizure(X_aug, y_aug, ratio=3)
        svm_model.fit(X_aug_os, y_aug_os)
        svm_model.prune_support_vectors(threshold=1e-3)
        decision_scores = svm_model.decision_function(X_test_scaled)
        y_pred_after = (decision_scores > 0.4).astype(int)

        # 후처리 재적용
        y_pred = y_pred_after.copy()
        for i in range(len(y_pred_after)):
            if y_pred_after[i] == 1:
                count = 1
                for j in range(1, 4):
                    if i + j < len(y_pred_after) and y_pred_after[i + j] == 1:
                        count += 1
                    else:
                        break
                if count < 4:
                    y_pred[i] = 0


    # 평가
    # Note: The original code re-predicted after online tuning, but then evaluated using svm_model.predict(X_test_scaled)
    # which is the model after tuning. The y_pred from post-processing should be used here.
    # Using y_pred (post-processed after tuning) for the report and other metrics.
    latencies = compute_latency_per_event(y_test, y_pred, step_sec=1)
    latency = compute_latency_in_event(y_test, y_pred, step_sec=1)
    vec_sens_60 = evaluate_vector_based_detection(y_test, y_pred, threshold=0.9)

    report_before = classification_report(y_test, y_pred_before, output_dict=True, zero_division=0)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"📏 Vector-based Sensitivity: {vec_sens_60*100:.1f}%" if vec_sens_60 is not None else "❌ No seizure event found.")
    print(f"⏱️ Latency: {latency:.5f} sec" if latency is not None else "❌ No latency computed.")

    print(f"📊 [환자: {patient_id}] 평가 결과:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(cm)

    one_shot_results.append({
        'patient': patient_id,
        'accuracy': report['accuracy'],
        'sensitivity': report['1.0']['recall'],
        'specificity': report['0.0']['recall'],
        'f1_seizure': report['1.0']['f1-score'],
        'latency': latency,
        'latencies' :latencies,
        'vec_sens_60': vec_sens_60
    })
    one_shot_results_before.append({
        'patient': patient_id,
        'accuracy': report_before['accuracy'],
        'sensitivity': report_before.get('1.0', {}).get('recall', 0.0),
        'specificity': report_before.get('0.0', {}).get('recall', 0.0),
        'f1_seizure': report_before.get('1.0', {}).get('f1-score', 0.0),
    })

    #모델 저장
    model_path = os.path.join(base_path, f"svm_model_{patient_id}.joblib")
    joblib.dump(svm_model, model_path)
    model_size_kb = os.path.getsize(model_path) / 1024

    t0 = time.time()
    _ = svm_model.decision_function(X_test_scaled[:100])  # 100개만 측정
    t1 = time.time()
    avg_pred_time = (t1 - t0) / 100

    # 전체 예측 처리 시간
    t2 = time.time()
    _ = svm_model.decision_function(X_test_scaled)
    t3 = time.time()
    total_test_time = t3 - t2

    # 특성 벡터 평균 메모리
    sample_feature_mem = X_test_scaled[0].nbytes  # 1개 샘플의 메모리 사용량

    # 저장
    one_shot_results[-1].update({
        'model_kb': round(model_size_kb, 2),
        'sample_feature_kb': round(sample_feature_mem / 1024, 3),
        'pred_time_s': round(avg_pred_time, 6),
        'testset_time_s': round(total_test_time, 3),
    })

    # Store data for the last patient for PCA visualization
    last_X_train_scaled = X_train_scaled
    last_y_train = y_train
    last_X_test_scaled = X_test_scaled
    last_y_test = y_test
    last_svm_model = svm_model

    # ✅ 메모리 해제
    del X_pat, y_pat, df_info_pat, X_train, y_train, X_test, y_test, svm_model, scaler
    if 'X_new' in locals(): del X_new, y_new
    if 'X_aug' in locals(): del X_aug, y_aug
    if 'X_aug_os' in locals(): del X_aug_os, y_aug_os
    gc.collect()

# 결과 정리
df_results_before = pd.DataFrame(one_shot_results_before)
df_results = pd.DataFrame(one_shot_results)
df_results.to_csv(f"{base_path}/online_tuned_results.csv", index=False)
df_results_before.to_csv(f"{base_path}/online_tuned_results_before.csv", index=False)

print("\n✅ 전체 평균 성능 이전:")
display(df_results_before[['accuracy', 'sensitivity', 'specificity', 'f1_seizure']].mean().round(4))
print("\n✅ 전체 평균 성능 이후:")
display(df_results[['accuracy', 'sensitivity', 'specificity', 'f1_seizure']].mean().round(4))
