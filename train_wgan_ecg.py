import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from ecg_generator import ECGGenerator
from ecg_critic import ECGCritic
from wgan_ecg_model import gradient_penalty

# ---------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("generated_ecg_samples", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# --------- Dummy ECG Data (for visible output) -------
def generate_dummy_ecg(n_samples=1000, seq_len=1250):
    t = np.linspace(0, 1, seq_len)
    data = []
    for _ in range(n_samples):
        ecg = (
            0.6 * np.sin(2 * np.pi * 5 * t) +
            0.2 * np.sin(2 * np.pi * 15 * t) +
            0.05 * np.random.randn(seq_len)
        )
        data.append(ecg)
    return np.array(data)

# ---------------------------------------------------
def train():
    latent_dim = 100
    epochs = 50
    batch_size = 32

    data = generate_dummy_ecg()
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

    generator = ECGGenerator(latent_dim).to(device)
    critic = ECGCritic().to(device)

    opt_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_C = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))

    lambda_gp = 10
    n_critic = 5

    g_losses, c_losses = [], []

    for epoch in range(epochs):
        for _ in range(n_critic):
            idx = np.random.randint(0, len(data), batch_size)
            real = data_tensor[idx]

            z = torch.randn(batch_size, latent_dim).to(device)
            fake = generator(z)

            loss_C = critic(fake).mean() - critic(real).mean()
            gp = gradient_penalty(critic, real, fake, device)
            loss_C += lambda_gp * gp

            opt_C.zero_grad()
            loss_C.backward()
            opt_C.step()

        z = torch.randn(batch_size, latent_dim).to(device)
        fake = generator(z)
        loss_G = -critic(fake).mean()

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        g_losses.append(loss_G.item())
        c_losses.append(loss_C.item())

        print(f"Epoch {epoch:03d} | C_loss: {loss_C.item():.4f} | G_loss: {loss_G.item():.4f}")

        # 🔹 SAVE ECG OUTPUT (external)
        if epoch % 10 == 0:
            sample = fake[0].detach().cpu().numpy()
            plt.figure(figsize=(10, 3))
            plt.plot(sample)
            plt.title(f"Synthetic ECG - Epoch {epoch}")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")
            plt.tight_layout()
            plt.savefig(f"generated_ecg_samples/ecg_epoch_{epoch}.png")
            plt.close()

    # 🔹 SAVE TRAINING LOSS CURVES
    plt.figure()
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(c_losses, label="Critic Loss")
    plt.legend()
    plt.title("WGAN Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("figures/training_losses.png")
    plt.close()

    torch.save(generator.state_dict(), "checkpoints/G_ecg_final.pt")
    print("✅ Training complete. External outputs saved.")

if __name__ == "__main__":
    train()
