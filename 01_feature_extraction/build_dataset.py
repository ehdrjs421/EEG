import os
import glob
import numpy as np
import pandas as pd
import re

from preprocess import preprocess_edf
from bandpass import extract_subband_energy
from tca_fe import extract_tca_features


# =========================
# Window / Context Config
# =========================
WINDOW_LEN_SEC = 2
STEP_LEN_SEC = 1
CONTEXT_WINDOW_SIZE = 3


# =========================
# Seizure Time Parser
# =========================
def get_seizure_times(summary_file_path, target_edf_file_name):
    seizures = []

    with open(summary_file_path, 'r') as f:
        content = f.read()

    file_sections = content.split("File Name: ")
    for section in file_sections:
        if target_edf_file_name in section:
            lines = section.strip().split('\n')
            for line in lines:
                if "Seizure Start Time" in line:
                    start = int(re.findall(r'\d+', line)[-1])
                elif "Seizure End Time" in line:
                    end = int(re.findall(r'\d+', line)[-1])
                    seizures.append((start, end))
            break

    return seizures


# =========================
# Dataset Builder
# =========================
def build_dataset(database_path):
    all_X, all_y, all_info = [], [], []

    patient_dirs = sorted(glob.glob(os.path.join(database_path, 'chb*')))

    for patient_dir in patient_dirs:
        patient_id = os.path.basename(patient_dir)
        summary_file = os.path.join(patient_dir, f"{patient_id}-summary.txt")
        edf_files = sorted(glob.glob(os.path.join(patient_dir, "*.edf")))

        print(f"\n📂 Processing {patient_id}")

        for edf_path in edf_files:
            edf_name = os.path.basename(edf_path)
            print(f"  🧠 {edf_name}")

            try:
                eeg, sfreq, n_channels = preprocess_edf(edf_path)
            except Exception as e:
                print(f"    ❌ Preprocess failed: {e}")
                continue

            # Sub-band spectral energy
            spectral_tensor = extract_subband_energy(
                eeg, sfreq,
                window_len_sec=WINDOW_LEN_SEC,
                step_len_sec=STEP_LEN_SEC
            )

            if spectral_tensor.shape[0] < CONTEXT_WINDOW_SIZE:
                continue

            # TCA-FE
            X_file, window_end_times = extract_tca_features(
                spectral_tensor,
                context_size=CONTEXT_WINDOW_SIZE,
                step_len_sec=STEP_LEN_SEC,
                window_len_sec=WINDOW_LEN_SEC
            )

            # Labeling
            seizure_times = get_seizure_times(summary_file, edf_name)
            y_file = np.zeros(len(X_file))

            for i, t_end in enumerate(window_end_times):
                t_start = t_end - WINDOW_LEN_SEC
                for s_start, s_end in seizure_times:
                    if max(t_start, s_start) < min(t_end, s_end):
                        y_file[i] = 1
                        break

            all_X.append(X_file)
            all_y.append(y_file)

            for idx in range(len(X_file)):
                all_info.append({
                    "patient": patient_id,
                    "file": edf_name,
                    "window_idx": idx,
                    "label": int(y_file[idx])
                })

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    df_info = pd.DataFrame(all_info)

    return X, y, df_info
