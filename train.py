
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from dotenv import load_dotenv
import kaggle

load_dotenv()

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# The set_experiment API creates a new experiment if it doesn't exist.
mlflow.set_experiment("Assignment3 Mohamed Hassan")

# IMPORTANT: Enable system metrics monitoring
mlflow.config.enable_system_metrics_logging()
mlflow.config.set_system_metrics_sampling_interval(1)


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# ──────────────────────────────────────────────
# Device Configuration
# ──────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("Random seed set to 42 for reproducibility.")

# you need to get your kaggle username and key from https://www.kaggle.com/settings/api
# and save them in a .env file in the same directory as this script
# .env file should look like this:
# KAGGLE_USERNAME = your_username
# KAGGLE_KEY = your_key


# Directories
data_dir = "data"
output_dir = "output"
os.makedirs(data_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

dataset_name = "mloey1/ahcd1"
csv_file_check = os.path.join(data_dir, "csvTrainImages 13440x1024.csv")

if not os.path.exists(csv_file_check):
    print("Dataset not found locally. Downloading from Kaggle...")
    kaggle.api.dataset_download_files(dataset_name, path=data_dir, unzip=True)
    print("Download and extraction complete!")
else:
    print(
        f"Dataset already found locally ({csv_file_check}). Skipping download.")

# ──────────────────────────────────────────────
# Imports and Data Prep
# ──────────────────────────────────────────────

df = pd.read_csv(csv_file_check)
data = torch.tensor(df.values, dtype=torch.float32).to(device) / 255.0

# ──────────────────────────────────────────────
# Model Architecture
# ──────────────────────────────────────────────

#  Generator


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.Sigmoid()  # for Generator output [0, 1]
        )

    def forward(self, x):
        return self.model(x)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameters
# Run 1: lr=0.0002, batch_size=64
# Run 2: lr=0.001,  batch_size=64
# Run 3: lr=0.0001, batch_size=64
# Run 4: lr=0.0002, batch_size=128
# Run 5: lr=0.0002, batch_size=32
# Run 6: lr=0.001, batch_size=64, epochs=20
# Run 7: lr=0.001, batch_size=64, epochs=150
# Run 8: lr=0.001, batch_size=64, epochs=30


LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 30
LATENT_DIM = 100

# ──────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────
with mlflow.start_run():

    # Log hyperparameters
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("latent_dim", LATENT_DIM)
    mlflow.log_param("optimizer", "Adam")

    # Log tags
    mlflow.set_tag("student_id", "202202037")
    mlflow.set_tag("model_type", "GAN")

    gen = Generator().to(device)
    disc = Discriminator().to(device)
    opt_gen = optim.Adam(
        gen.parameters(),
        lr=LEARNING_RATE,
        betas=(
            0.5,
            0.999))
    opt_disc = optim.Adam(
        disc.parameters(),
        lr=LEARNING_RATE,
        betas=(
            0.5,
            0.999))
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        epoch_real_correct = epoch_fake_correct = epoch_samples = 0
        epoch_lossD = epoch_lossG = 0.0
        num_batches = 0

        for i in range(0, len(data), BATCH_SIZE):
            real_batch = data[i:i + BATCH_SIZE]
            curr_batch_size = len(real_batch)
            epoch_samples += curr_batch_size

            # Train Discriminator
            opt_disc.zero_grad()
            disc_real = disc(real_batch).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            epoch_real_correct += (disc_real > 0).sum().item()

            noise = torch.randn(curr_batch_size, LATENT_DIM, device=device)
            fake_batch = gen(noise).detach()
            disc_fake = disc(fake_batch).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            epoch_fake_correct += (disc_fake < 0).sum().item()

            lossD = (lossD_real + lossD_fake) / 2
            lossD.backward()
            opt_disc.step()

            # Train Generator
            opt_gen.zero_grad()
            output = disc(gen(noise)).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            lossG.backward()
            opt_gen.step()

            epoch_lossD += lossD.item()
            epoch_lossG += lossG.item()
            num_batches += 1

        disc_acc = (epoch_real_correct + epoch_fake_correct) / \
            (2 * epoch_samples) * 100
        avg_lossD = epoch_lossD / num_batches
        avg_lossG = epoch_lossG / num_batches

        # Log metrics every epoch
        mlflow.log_metric("discriminator_loss", avg_lossD, step=epoch)
        mlflow.log_metric("generator_loss", avg_lossG, step=epoch)
        mlflow.log_metric("discriminator_accuracy", disc_acc, step=epoch)

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(
                f"Epoch {epoch:3d} | D Loss: {avg_lossD:.4f} | G Loss: {avg_lossG:.4f} | D Acc: {disc_acc:.2f}%")

    # Save generated samples image
    with torch.no_grad():
        test_noise = torch.randn(5, LATENT_DIM, device=device)
        generated_imgs = gen(test_noise).cpu().detach().numpy()

    plt.figure(figsize=(12, 3))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(generated_imgs[i].reshape(32, 32), cmap='gray')
        plt.title(f"Sample {i + 1}")
        plt.axis('off')
    plt.tight_layout()
    output_img_path = os.path.join(output_dir, "generated_samples.png")
    plt.savefig(output_img_path)
    plt.close()
    mlflow.log_artifact(output_img_path)

# Save generated samples image
    with torch.no_grad():
        test_noise = torch.randn(5, LATENT_DIM, device=device)
        generated_imgs = gen(test_noise).cpu().detach().numpy()

    plt.figure(figsize=(12, 3))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(generated_imgs[i].reshape(32, 32), cmap='gray')
        plt.title(f"Sample {i + 1}")
        plt.axis('off')
    plt.tight_layout()
    output_img_path = os.path.join(output_dir, "generated_samples.png")
    plt.savefig(output_img_path)
    plt.close()
    mlflow.log_artifact(output_img_path)

    # Save Models
    gen.eval()
    disc.eval()

    from mlflow.models.signature import infer_signature

    with torch.no_grad():
        dummy_input = torch.randn(1, LATENT_DIM, device=device)
        dummy_output = gen(dummy_input).cpu().numpy()
    dummy_input_np = dummy_input.cpu().numpy()
    signature = infer_signature(dummy_input_np, dummy_output)

    # Define conda env manually to avoid the +cu128 warning
    conda_env = {
        "channels": ["defaults"],
        "dependencies": [
            "python=3.10",
            "pip",
            {"pip": ["mlflow", "torch==2.10.0", "numpy"]}
        ],
        "name": "mlflow-env"
    }

    # Log generator ONCE with full metadata
    mlflow.pytorch.log_model(
        gen,
        name="generator_model",
        signature=signature,
        input_example=dummy_input_np,
        conda_env=conda_env
    )

    # Log discriminator ONCE
    mlflow.pytorch.log_model(
        disc,
        name="discriminator_model",
        conda_env=conda_env
    )

    # Backup raw weights
    gen_path = os.path.join(output_dir, "generator.pth")
    disc_path = os.path.join(output_dir, "discriminator.pth")
    torch.save(gen.state_dict(), gen_path)
    torch.save(disc.state_dict(), disc_path)
    mlflow.log_artifact(gen_path)
    mlflow.log_artifact(disc_path)

    # Log environment.yml and model metadata as a directory artifact
    import tempfile
    import shutil
    import json
    env_path = os.path.abspath("environment.yml")
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Copy environment.yml if it exists
        if os.path.exists(env_path):
            shutil.copy(env_path, os.path.join(tmp_dir, "environment.yml"))
        # Save model metadata as JSON
        metadata = {
            "student_id": "202202037",
            "model_type": "GAN",
            "latent_dim": LATENT_DIM,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE
        }
        with open(os.path.join(tmp_dir, "model_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        # Log the directory as an artifact
        mlflow.log_artifacts(tmp_dir, artifact_path="run_env_and_metadata")
