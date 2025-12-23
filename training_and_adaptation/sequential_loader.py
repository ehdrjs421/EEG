import os
import numpy as np
import pandas as pd

def load_patient_data(base_path, patient_id):
    try:
        X = np.load(os.path.join(base_path, f"X_{patient_id}.npy")).astype(np.float32)
        y = np.load(os.path.join(base_path, f"y_{patient_id}.npy"))
        # X = X.replace('\\', '/')
        # y = y.replace('\\', '/')
        df_info = pd.read_pickle(
            os.path.join(base_path, f"df_{patient_id}.pkl")
        ).reset_index(drop=True)
        return X, y, df_info
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {patient_id} ({e})")
        return None, None, None
