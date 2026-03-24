import os
import glob
import re
import numpy as np
import pandas as pd
import mne

from feature_extraction.preprocess import load_and_preprocess_edf
from feature_extraction.tca_fe import extract_tca_features


def get_seizure_times(summary_file_path, target_edf_file_name):
    seizures = []
    start_time = None

    with open(summary_file_path, 'r') as f:
        lines = f.readlines()

    in_file = False
    for line in lines:
        if target_edf_file_name in line:
            in_file = True
        elif in_file and line.strip() == "":
            break

        if in_file:
            if re.match(r"Seizure\s+\d+\s+Start Time:", line):
                start_time = int(line.split(":")[-1].strip().split()[0])
            elif re.match(r"Seizure\s+\d+\s+End Time:", line):
                end_time = int(line.split(":")[-1].strip().split()[0])
                seizures.append((start_time, end_time))
            elif "Seizure Start Time:" in line:
                start_time = int(line.split(":")[-1].strip().split()[0])
            elif "Seizure End Time:" in line:
                end_time = int(line.split(":")[-1].strip().split()[0])
                seizures.append((start_time, end_time))

    return seizures

TARGET_SFREQ = 128  # Hz, 논문 기반
CHANNELS_TO_USE = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
                   'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                   'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
                   'FP2-F8', 'F8-T8', 'FT10-T8', 'P8-O2'] # 예시 16 채널
N_CHANNELS = len(CHANNELS_TO_USE)

SUB_BANDS = [(0.5, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28)]
N_SUBBANDS = len(SUB_BANDS)

WINDOW_LEN_SEC = 2  # 2초 윈도우
STEP_LEN_SEC = 1    # 1초 간격 (핑퐁 전략)
WINDOW_LEN_SAMPLES = int(WINDOW_LEN_SEC * TARGET_SFREQ)
STEP_LEN_SAMPLES = int(STEP_LEN_SEC * TARGET_SFREQ)

CONTEXT_WINDOW_SIZE = 3 # TCA-FE를 위한 컨텍스트 윈도우 (3개의 2초 윈도우)

def build_dataset(
    edf_root,
    summary_root,
    channels_to_use = CHANNELS_TO_USE,
    target_sfreq = TARGET_SFREQ,
    sub_bands = SUB_BANDS,
    window_len_samples =WINDOW_LEN_SAMPLES,
    step_len_samples = STEP_LEN_SAMPLES,
    context_window_size =CONTEXT_WINDOW_SIZE,
    target_patients=None  # 특정 환자만 지정 (예: ['chb23', 'chb24'])
):
    """
    Wrapper of EEG2.py feature + label extraction pipeline.
    Original logic preserved.
    """

    all_patients_info = []

    patient_dirs = sorted(glob.glob(os.path.join(edf_root, "chb*")))

    TARGET_PATIENTS = {'chb01', 'chb02', 'chb03', 'chb04'}
    patient_dirs = [p for p in sorted(glob.glob(os.path.join(edf_root, "chb*")))
                    if os.path.basename(p) in TARGET_PATIENTS]

    EXPECTED_FEATURE_SIZE = None
    for patient_dir in patient_dirs:
        patient_id = os.path.basename(patient_dir)
        
        # 지정된 환자만 처리 (입력된 경우)
        if target_patients is not None and patient_id not in target_patients:
            continue
            
        # # 데이터 처리 분리 진행
        print(f"{patient_id}")
        summary_file1 = os.path.join(edf_root, f"{patient_id}")
        summary_file = os.path.join(summary_file1, f"{patient_id}-summary.txt")
        summary_file = summary_file.replace('\\', '/')

        edf_files = sorted(glob.glob(os.path.join(patient_dir, "*.edf")))
        current_patient_features = []
        current_patient_labels = []
        current_patient_info = []
        
        for edf_path in edf_files:
            print(edf_path)
            edf_name = os.path.basename(edf_path)
            # if not ('chb01_01.edf' <= edf_name <= 'chb01_10.edf'):
                # continue
            print(edf_name)

            # ===============================
            # Load EEG
            # ===============================
            eeg_data, sfreq = load_and_preprocess_edf(
                edf_path,
                channels_to_use,
                target_sfreq
            )

            if eeg_data is None:
                continue

            # ===============================
            # TCA Feature Extraction
            # ===============================
            features, end_times = extract_tca_features(
                eeg_data,
                sfreq,
                sub_bands,
                window_len_samples,
                step_len_samples,
                context_window_size
            )

            if features is None:
                continue

            final_file_features = features
            
            if EXPECTED_FEATURE_SIZE is None:
                EXPECTED_FEATURE_SIZE = final_file_features.shape[1]
            elif final_file_features.shape[1] != EXPECTED_FEATURE_SIZE:
                print(f"    ⚠️ Skipping {edf_name}: feature size mismatch. Got {final_file_features.shape[1]}, expected {EXPECTED_FEATURE_SIZE}.")
                continue

            # ===============================
            # Seizure time parsing
            # ===============================
            seizure_periods = get_seizure_times(summary_file, edf_name)

            file_labels = np.zeros(len(features))

            for k, end_idx in enumerate(end_times):
                feature_vec_end_time = end_idx * step_len_samples / sfreq
                feature_vec_start_time = feature_vec_end_time - (
                    window_len_samples / sfreq
                )

                is_seizure = False
                for seizure_start, seizure_end in seizure_periods:
                    # ✨ pre-ictal window 적용
                    # 발작 시작 30초 전부터 발작 종료까지 양성
                    pre_ictal_start = max(0, seizure_start - PRE_ICTAL_SEC)
                    overlap_start = max(feature_vec_start_time, pre_ictal_start)
                    overlap_end   = min(feature_vec_end_time,   seizure_end)
                    if overlap_start < overlap_end:
                        is_seizure = True
                        break

                if is_seizure:
                    file_labels[k] = 1

            # ===============================
            # Accumulate
            # ===============================

            for win_idx in range(len(file_labels)):
                all_patients_info.append({
                    "patient": patient_id,
                    "file": edf_name,
                    "window_index_in_file": win_idx,
                    "label": file_labels[win_idx]
                })
                
            current_patient_features.append(features)
            current_patient_labels.append(file_labels)
            for win_idx in range(len(file_labels)):
                current_patient_info.append({
                    "patient": patient_id,
                    "file": edf_name,
                    "window_index_in_file": win_idx,
                    "label": file_labels[win_idx]
                })
        if current_patient_features:
            X_patient = np.concatenate(current_patient_features, axis=0)
            y_patient = np.concatenate(current_patient_labels, axis=0)
            df_info_patient = pd.DataFrame(current_patient_info)

            # 파일명 설정 (예: X_chb01.npy, y_chb01.npy, info_chb01.csv)
            save_path_X = os.path.join(summary_root, f"X_{patient_id}.npy").replace('\\', '/')
            save_path_y = os.path.join(summary_root, f"y_{patient_id}.npy").replace('\\', '/')
            save_path_info = os.path.join(summary_root, f"df_{patient_id}.pkl").replace('\\', '/')

            np.save(save_path_X, X_patient)
            np.save(save_path_y, y_patient)
            df_info_patient.to_pickle(save_path_info)

            print(f"💾 Saved {patient_id} data to {summary_root}")
        else:
            print(f"⚠️ No data collected for {patient_id}")

    if not all_patients_info:
        raise RuntimeError("No features were extracted from any file.")

    # 메모리를 많이 차지하는 전체 X, y는 반환하지 않음 (Colab RAM 초과 방지)
    df_info = pd.DataFrame(all_patients_info)

    return np.array([]), np.array([]), df_info
