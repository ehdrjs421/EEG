from scipy.signal import butter, lfilter

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if low <= 0:
        b, a = butter(order, high, btype='low')
    elif high >= 1:
        b, a = butter(order, low, btype='high')
    else:
        b, a = butter(order, [low, high], btype='band')

    return lfilter(b, a, data)
