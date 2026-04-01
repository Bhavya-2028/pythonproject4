import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut=0.5, highcut=40, fs=250, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def normalize(signal):
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

def segment_signal(signal, segment_length=1250):
    segments = []
    for i in range(0, len(signal) - segment_length, segment_length):
        segments.append(signal[i:i+segment_length])
    return np.array(segments)