import streamlit as st
import matplotlib.pyplot as plt
from inference_ecg import load_model, generate_ecg

model = load_model("checkpoints/G_epoch_180.pt")

st.title("🫀 WGAN ECG Generator")

n_samples = st.slider("Number of Samples", 1, 10, 3)

if st.button("Generate ECG"):
    samples = generate_ecg(model, n_samples)
    for s in samples:
        fig, ax = plt.subplots()
        ax.plot(s)
        st.pyplot(fig)