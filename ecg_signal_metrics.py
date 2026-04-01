import numpy as np
from scipy.signal import welch

def compute_psd(signal, fs=250):
    f, pxx = welch(signal, fs)
    return f, pxx

def basic_stats(signal):
    return {
        "mean": np.mean(signal),
        "variance": np.var(signal),
        "max": np.max(signal),
        "min": np.min(signal)
    }