import os
import time
import joblib


def analyze_model_resources(
    model,
    X_test,
    save_path,
    sample_size=100
):
    joblib.dump(model, save_path)
    model_kb = os.path.getsize(save_path) / 1024

    t0 = time.time()
    _ = model.decision_function(X_test[:sample_size])
    t1 = time.time()
    avg_pred_time = (t1 - t0) / sample_size

    t2 = time.time()
    _ = model.decision_function(X_test)
    t3 = time.time()
    total_test_time = t3 - t2

    sample_feature_kb = X_test[0].nbytes / 1024

    return {
        'model_kb': round(model_kb, 2),
        'sample_feature_kb': round(sample_feature_kb, 3),
        'pred_time_s': round(avg_pred_time, 6),
        'testset_time_s': round(total_test_time, 3)
    }
