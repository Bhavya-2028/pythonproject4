import matplotlib.pyplot as plt

def plot_ecg(signal, title="ECG Signal"):
    plt.figure(figsize=(10,3))
    plt.plot(signal)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()