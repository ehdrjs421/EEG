import numpy as np
import mne

AMPLITUDE_THRESHOLD_UV = 500 

def load_and_preprocess_edf(
    edf_path,
    channels_to_use,
    target_sfreq
):
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except (ValueError, EOFError) as e:
        # Bad EDF file 오류나 파일 끝 에러를 여기서 잡음
        print(f"    ❌ Bad EDF file (Corrupted): {edf_path}")
        return None, None
    except Exception as e:
        print(f"    ❌ Error loading {edf_path}: {e}")
        return None, None    
    
    # raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # 채널명 정리
    raw.rename_channels(lambda s: s.strip().replace('.', ''))

    # 채널 선택
    available_channels = [ch for ch in channels_to_use if ch in raw.ch_names]
    if len(available_channels) < len(channels_to_use):
        if len(available_channels) < 8:
            return None, None
        raw.pick(available_channels)
    else:
        raw.pick(channels_to_use)

    # 리샘플링
    raw.resample(target_sfreq, verbose=False)
    data = raw.get_data()
    # ✨ 진폭 임계값 기반 아티팩트 제거
    # V → μV 변환 후 임계값 적용
    data_uv = data * 1e6
 
    # 임계값 초과 구간을 채널 평균으로 대체
    # (0으로 채우면 발작 신호 경계에서 왜곡 발생 가능)
    artifact_mask = np.abs(data_uv) > AMPLITUDE_THRESHOLD_UV
    n_artifact = artifact_mask.sum()
 
    if n_artifact > 0:
        for ch_idx in range(data_uv.shape[0]):
            ch_mask = artifact_mask[ch_idx]
            if ch_mask.any():
                ch_mean = data_uv[ch_idx, ~ch_mask].mean() if (~ch_mask).any() else 0.0
                data_uv[ch_idx, ch_mask] = ch_mean
 
        artifact_ratio = n_artifact / data_uv.size
        if artifact_ratio > 0.01:  # 1% 이상이면 경고
            print(f"    ⚠️ Artifact detected: {n_artifact} samples "
                  f"({artifact_ratio*100:.2f}%) replaced")
 
    # μV → V 복원
    data_clean = data_uv / 1e6
 
    return data_clean, raw.info['sfreq']
    # return raw.get_data(), raw.info['sfreq']
