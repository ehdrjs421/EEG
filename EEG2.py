import mne
import os
import glob
import re
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
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

# --- HELPER FUNCTIONS FOR FEATURE EXTRACTION ---
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if low <= 0: # Ensure lowcut is positive
        b, a = butter(order, high, btype='low')
    elif high >= 1: # Ensure highcut is less than Nyquist
        b, a = butter(order, low, btype='high')
    else:
        b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def get_seizure_times(summary_file_path, target_edf_file_name):
    seizures = []
    try:
        with open(summary_file_path, 'r') as f:
            content = f.read()

        file_sections = content.split("File Name: ")
        for section in file_sections:
            if target_edf_file_name in section:
                lines = section.strip().split('\n')
                for line in lines:
                    if re.match(r"Seizure\s+\d+\s+Start Time:", line):
                        start_time = int(re.findall(r'\d+', line)[-1])
                    elif re.match(r"Seizure\s+\d+\s+End Time:", line):
                        end_time = int(re.findall(r'\d+', line)[-1])
                        seizures.append((start_time, end_time))
                    elif "Seizure Start Time:" in line:  # chb24 등 단일 형식
                        start_time = int(re.findall(r'\d+', line)[0])
                    elif "Seizure End Time:" in line:
                        end_time = int(re.findall(r'\d+', line)[0])
                        seizures.append((start_time, end_time))
                break  # 원하는 파일 찾은 경우
    except FileNotFoundError:
        print(f"❌ Summary file not found at: {summary_file_path}")
    return seizures

# --- MAIN PROCESSING LOOP FOR FEATURE EXTRACTION ---
all_patients_features = []
all_patients_labels = []
all_patients_info = [] # (환자 ID, 파일명, 윈도우 인덱스 등 추적용)

# CHB-MIT 데이터베이스 경로 (사용자 환경에 맞게 수정)
database_path = '/content/drive/My Drive/chb-mit-scalp-eeg-database-1.0.0/'
patient_dirs = sorted(glob.glob(os.path.join(database_path, 'chb*')))  # chb01 ~ chb24 폴더

EXPECTED_FEATURE_SIZE = None
for patient_folder in patient_dirs:
    patient_id = os.path.basename(patient_folder)
    summary_file = os.path.join(patient_folder, f"{patient_id}-summary.txt")
    edf_files = sorted(glob.glob(os.path.join(patient_folder, '*.edf')))

    print(f"\n📂 Processing {patient_id} with {len(edf_files)} EDF files")

    for edf_path in edf_files:
        edf_file_name = os.path.basename(edf_path)
        print(f"  🧠 File: {edf_file_name}")

        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        except Exception as e:
            print(f"    ❌ Error loading {edf_file_name}: {e}")
            continue

        # 0. 채널 이름에서 공백이나 불필요한 문자 제거
        raw.rename_channels(lambda s: s.strip().replace('.', ''))

        # 1. 채널 선택 (CHANNELS_TO_USE에 있는 채널만 선택)
        available_channels = [ch for ch in CHANNELS_TO_USE if ch in raw.ch_names]
        if len(available_channels) < N_CHANNELS:
            print(f"    Warning: Not all {N_CHANNELS} desired channels found in {edf_file_name}. Found {len(available_channels)}. Skipping file or using available.")
            if len(available_channels) < 8: # 예를 들어 최소 8개 채널은 있어야 한다고 가정
                print(f"    Skipping {edf_file_name} due to insufficient common channels.")
                continue
            current_n_channels = len(available_channels)
            raw.pick(available_channels)
        else:
            raw.pick(CHANNELS_TO_USE)
            current_n_channels = N_CHANNELS

        # 2. 리샘플링
        raw.resample(TARGET_SFREQ, verbose=False)
        eeg_data = raw.get_data() # (n_channels, n_times)
        sfreq = raw.info['sfreq']

        # 3. 스펙트럼 에너지 특징 추출
        file_spectral_features_list = [] # 각 채널, 각 서브밴드에 대한 특징 리스트
        for ch_idx in range(current_n_channels):
            channel_data = eeg_data[ch_idx, :]
            channel_sub_band_energy_list = []
            for lowcut, highcut in SUB_BANDS:
                filtered_data = bandpass_filter(channel_data, lowcut, highcut, sfreq)

                sub_band_spectral_energy_for_windows = []
                for i in range(0, len(filtered_data) - WINDOW_LEN_SAMPLES + 1, STEP_LEN_SAMPLES):
                    window = filtered_data[i : i + WINDOW_LEN_SAMPLES]
                    spectral_energy = np.sum(np.abs(window))
                    sub_band_spectral_energy_for_windows.append(spectral_energy)
                channel_sub_band_energy_list.append(sub_band_spectral_energy_for_windows)
            file_spectral_features_list.append(channel_sub_band_energy_list)

        if not file_spectral_features_list or not file_spectral_features_list[0] or not file_spectral_features_list[0][0]:
            print(f"    No spectral features extracted for {edf_file_name}. Skipping.")
            continue

        # 모든 채널/서브밴드에 대해 동일한 윈도우 수를 갖도록 패딩 또는 자르기
        min_windows_in_file = min(len(sb_energy) for ch_energy in file_spectral_features_list for sb_energy in ch_energy)

        if min_windows_in_file < CONTEXT_WINDOW_SIZE : # 컨텍스트 윈도우를 만들 수 없을 정도로 짧으면 스킵
            print(f"    File {edf_file_name} too short ({min_windows_in_file} windows) to form context. Skipping.")
            continue

        # (n_channels, n_subbands, n_windows_in_file) 형태로 정렬
        spectral_energy_tensor_file = np.array(
            [[sb_energy[:min_windows_in_file] for sb_energy in ch_energy] for ch_energy in file_spectral_features_list]
        )
        # (n_windows_in_file, n_channels, n_subbands) 형태로 transpose
        spectral_energy_tensor_file = spectral_energy_tensor_file.transpose(2, 0, 1)

        # 4. TCA-FE 특징 생성
        file_ta_features = []
        file_ca_features = []
        window_end_times_sec = [] # 각 특징 벡터가 끝나는 시간 (초 단위)

        for i in range(spectral_energy_tensor_file.shape[0] - CONTEXT_WINDOW_SIZE + 1):
            pre_tca_block = spectral_energy_tensor_file[i : i + CONTEXT_WINDOW_SIZE, :, :]

            ta_feature = np.mean(pre_tca_block, axis=0) # (n_channels, n_subbands)
            file_ta_features.append(ta_feature.flatten()) # 1D (n_channels * n_subbands)

            ca_feature = np.mean(pre_tca_block, axis=1) # (CONTEXT_WINDOW_SIZE, n_subbands)
            file_ca_features.append(ca_feature.flatten()) # 1D (CONTEXT_WINDOW_SIZE * n_subbands)

            current_window_end_time = (i + CONTEXT_WINDOW_SIZE -1) * STEP_LEN_SEC + WINDOW_LEN_SEC
            window_end_times_sec.append(current_window_end_time)


        if not file_ta_features:
            print(f"    No TCA features generated for {edf_file_name}. Skipping.")
            continue

        final_file_features = np.hstack((np.array(file_ta_features), np.array(file_ca_features)))

        if EXPECTED_FEATURE_SIZE is None:
            EXPECTED_FEATURE_SIZE = final_file_features.shape[1]
        elif final_file_features.shape[1] != EXPECTED_FEATURE_SIZE:
            print(f"    ⚠️ Skipping {edf_file_name}: feature size mismatch. Got {final_file_features.shape[1]}, expected {EXPECTED_FEATURE_SIZE}.")
            continue

        # 5. 레이블 생성
        seizure_periods = get_seizure_times(summary_file, edf_file_name)
        file_labels = np.zeros(len(final_file_features))

        for k, feature_vec_end_time in enumerate(window_end_times_sec):
            feature_vec_start_time = feature_vec_end_time - WINDOW_LEN_SEC # 현재 윈도우의 시작 시간
            is_seizure = False
            for seizure_start, seizure_end in seizure_periods:
                # 윈도우가 발작 구간과 50% 이상 겹치면 발작으로 간주 (예시 기준)
                overlap_start = max(feature_vec_start_time, seizure_start)
                overlap_end = min(feature_vec_end_time, seizure_end)
                if overlap_end > overlap_start: # 겹치는 경우
                    overlap_duration = overlap_end - overlap_start
                    if overlap_duration >= 0:
                        is_seizure = True
                        break
            if is_seizure:
                file_labels[k] = 1

        all_patients_features.append(final_file_features)
        all_patients_labels.append(file_labels)
        for win_idx in range(len(final_file_features)):
             all_patients_info.append({'patient': patient_id, 'file': edf_file_name, 'window_index_in_file': win_idx, 'label': file_labels[win_idx]})


# 모든 환자/파일 데이터 취합
if all_patients_features:
    X = np.concatenate(all_patients_features, axis=0)
    y = np.concatenate(all_patients_labels, axis=0)
    df_info = pd.DataFrame(all_patients_info)
else:
    print("No features were extracted from any file. Exiting.")
    exit()

print(f"\nTotal features extracted: {X.shape[0]}, Number of features per sample: {X.shape[1]}")
print(f"Label distribution: {pd.Series(y).value_counts(normalize=True)}")
