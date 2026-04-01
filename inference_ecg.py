import torch
import numpy as np
from ecg_generator import ECGGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path, latent_dim=100):
    model = ECGGenerator(latent_dim)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def generate_ecg(model, n_samples=1, latent_dim=100):
    z = torch.randn(n_samples, latent_dim).to(device)
    with torch.no_grad():
        fake = model(z).cpu().numpy()
    return fake