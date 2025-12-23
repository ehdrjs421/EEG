import mne

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
    
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

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

    return raw.get_data(), raw.info['sfreq']
