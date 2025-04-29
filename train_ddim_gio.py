# train_ddim.py

import os
import sys
import subprocess

import torch
import time
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import UNet2DModel, DDPMScheduler
from tqdm import tqdm




# Parametri di configurazione
DATA_DIR     = "128x128_images"
OUTPUT_DIR   = "./ddim_model"
BATCH_SIZE   = 16
IMAGE_SIZE   = 128
IN_CHANNELS  = 3
NUM_TIMESTEPS= 1000
LEARNING_RATE= 1e-4
EPOCHS       = 2

# Definisce device (GPU se disponibile, altrimenti CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Funzione di normalizzazione [-1,1]
def normalize_to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    return 2.0 * x - 1.0


# Dataset custom che esplora sottocartelle
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.transform = transform
        self.paths = []
        for subdir, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                    self.paths.append(os.path.join(subdir, f))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


# Loss MSE per predire direttamente il rumore
mse = nn.MSELoss()

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    scheduler: DDPMScheduler,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    model.train()
    total_loss = 0.0

    for imgs in tqdm(dataloader, desc="Training"):
        imgs = imgs.to(device)

        # timesteps casuali
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (imgs.size(0),),
            device=device
        ).long()

        # genera e aggiunge rumore
        noise = torch.randn_like(imgs)
        noisy = scheduler.add_noise(imgs, noise, timesteps)

        # forward
        preds = model(noisy, timesteps).sample

        # loss predicendo il rumore vero
        loss = mse(preds, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


if __name__ == "__main__":
    
    script_start = time.perf_counter()


    # Check ambiente prima di tutto
    print(f"\nTraining on device: {DEVICE}\n")

    # Prepara cartella di output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Trasformazioni (senza lambda anonima)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(normalize_to_minus1_1)
    ])

    # Dataset e DataLoader
    dataset = ImageFolderDataset(DATA_DIR, transform=transform)
    dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,           # 8–12 workers se la CPU e i dischi lo reggono
    pin_memory=True,         # trasferimento GPU più rapido
    persistent_workers=True, # mantieni vivi i worker
    drop_last=True
    )


    # Costruisci modello e scheduler
    model = UNet2DModel(
        sample_size=IMAGE_SIZE,
        in_channels=IN_CHANNELS,
        out_channels=IN_CHANNELS,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D",)*4,
        up_block_types=("UpBlock2D",)*4
    ).to(DEVICE)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=NUM_TIMESTEPS,
        beta_schedule="squaredcos_cap_v2"
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Ciclo di addestramento
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_epoch(model, dataloader, noise_scheduler, optimizer, DEVICE)
        print(f"Epoch {epoch:03d} — avg loss: {avg_loss:.5f}")

        if epoch % 10 == 0:
            ckpt = os.path.join(OUTPUT_DIR, f"ddim_epoch{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss
            }, ckpt)
            print(f"Checkpoint salvato: {ckpt}")

    # Salvataggio finale
    final = os.path.join(OUTPUT_DIR, "ddim_final.pt")
    torch.save(model.state_dict(), final)
    print(f"\nAddestramento completato. Modello salvato in: {final}")

    script_end = time.perf_counter()
    elapsed = script_end - script_start
    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print(f"\nTempo totale di esecuzione: {int(hrs)}h {int(mins)}m {secs:.1f}s")