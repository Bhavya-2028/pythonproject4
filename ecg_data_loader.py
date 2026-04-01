import numpy as np
import os

def load_numpy_ecg(data_path):
    """
    Load preprocessed ECG numpy file
    Expected shape: (num_samples, seq_len)
    """
    data = np.load(data_path)
    return data

def train_val_test_split(data, train_ratio=0.7, val_ratio=0.15):
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    return train, val, test