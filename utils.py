import os
import torch
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import os
import math
from PIL import Image, ImageEnhance
import numpy as np

NUM_TRAIN_TIMESTEPS = 1000


# DATA AUGMENTATION
def rotate_fixed(img: Image.Image, angle: float, size=128):
    rot = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=0)
    # ricentra su canvas 128×128
    canvas = Image.new("L", (size, size), 0)
    x = (size - rot.width) // 2
    y = (size - rot.height) // 2
    canvas.paste(rot, (x, y))
    return canvas


def horizontal_flip(img: Image.Image):
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def add_gaussian_noise(img: Image.Image, mean=0.0, std=10.0):
    arr = np.array(img).astype(np.float32)
    noise = np.random.RandomState(0).normal(mean, std, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def add_salt_pepper(img: Image.Image, prob=0.02):
    arr = np.array(img)
    rng = np.random.RandomState(1)
    h, w = arr.shape
    sp = arr.copy()
    # salt
    num_salt = int(math.ceil(prob * h * w * 0.5))
    coords = (rng.randint(0, h, num_salt), rng.randint(0, w, num_salt))
    sp[coords] = 255
    # pepper
    num_pepper = int(math.ceil(prob * h * w * 0.5))
    coords = (rng.randint(0, h, num_pepper), rng.randint(0, w, num_pepper))
    sp[coords] = 0
    return Image.fromarray(sp)


def change_brightness(img: Image.Image, factor=1.2):
    return ImageEnhance.Brightness(img).enhance(factor)


def change_contrast(img: Image.Image, factor=1.3):
    return ImageEnhance.Contrast(img).enhance(factor)

AUGMENTATION_FUNCTIONS = {
    "rot": lambda img, param, size: rotate_fixed(img, param, size),
    "flip": lambda img, param, size: horizontal_flip(img),
    "noise_gaussian": lambda img, param, size: add_gaussian_noise(
        img, **param
    ).resize((size, size), Image.BICUBIC),
    "noise_salt_pepper": lambda img, param, size: add_salt_pepper(
        img, **param
    ).resize((size, size), Image.BICUBIC),
    "bright": lambda img, param, size: change_brightness(img, **param).resize(
        (size, size), Image.BICUBIC
    ),
    "contrast": lambda img, param, size: change_contrast(img, **param).resize(
        (size, size), Image.BICUBIC
    ),
}


class AugmentedDataset(Dataset):

    def __init__(self, root_dir, image_size=128, augmentations=None):
        self.base_ds = datasets.ImageFolder(
            root=root_dir,
            transform=transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((image_size, image_size)),
                ]
            ),
        )

        self.image_size = image_size
        self.to_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

        # Default augmentation set
        if augmentations is None:
            self.augmentations = [
                {"type": None}, # no augmentation
                {"type": "rot", "param": -5}, 
                {"type": "rot", "param": 5},
                {"type": "flip"},
                {"type": "noise_gaussian", "param": {"mean": 0.0, "std": 10.0}},
                {"type": "noise_salt_pepper", "param": {"prob": 0.02}},
                {"type": "bright", "param": {"factor": 1.2}},
                {"type": "contrast", "param": {"factor": 1.3}},
            ]
        else:
            self.augmentations = augmentations

    def __len__(self):
        return len(self.base_ds) * len(self.augmentations)

    def __getitem__(self, idx):
        img_idx = idx // len(self.augmentations)
        aug_idx = idx % len(self.augmentations)

        img, label = self.base_ds[img_idx]
        aug = self.augmentations[aug_idx]

        if aug["type"] is not None:
            func = AUGMENTATION_FUNCTIONS[aug["type"]]
            param = aug.get("param", {})
            img = func(img, param, self.image_size)

        img_t = self.to_tensor(img)
        return img_t, label


def sample_images_from_pure_noise(
    output_path="result/ddim_sample.png",
    num_steps=NUM_TRAIN_TIMESTEPS,
    DEVICE="cpu",
    IMAGE_SIZE=128,
    model=None,
    ddim_scheduler=None,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    with torch.no_grad():
        # Sample random noise
        sample = torch.randn((1, 1, IMAGE_SIZE, IMAGE_SIZE), device=DEVICE)

        for t in tqdm(
            reversed(range(0, num_steps)), desc="Sampling DDIM", total=num_steps
        ):
            noise_pred = model(sample, torch.tensor([t], device=DEVICE)).sample
            sample = ddim_scheduler.step(noise_pred, t, sample).prev_sample

        # denormalize and save
        final = (sample.clamp(-1, 1) + 1) / 2
        transforms.ToPILImage()(final.squeeze(0).squeeze(0).cpu()).save(output_path)
        print(f"Sample saved to {output_path}")

def sample_images_from_validation(
    model,
    noise_scheduler,
    test_loader,
    output_dir="result/val_reconstructions",
    num_timesteps=NUM_TRAIN_TIMESTEPS,
    device="cpu",
    max_examples=5,
):
    """
    Per ogni batch in test_loader:
      • prende un'immagine pulita x0
      • aggiunge rumore x_t = q(x_t|x0) con t casuale
      • predice l'epsilon con la U-Net
      • ricostruisce x0_hat da x_t e epsilon_pred
      • salva side-by-side (clean | recon) per i primi max_examples
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    to_pil = transforms.ToPILImage()

    saved = 0
    with torch.no_grad():
        for clean_imgs, _ in tqdm(test_loader, desc="Reconstructions"):
            clean_imgs = clean_imgs.to(device)
            batch_size = clean_imgs.size(0)

            # rumore e timestep casuali
            noise = torch.randn_like(clean_imgs)
            timesteps = torch.randint(0, num_timesteps, (batch_size,), device=device)

            # noising
            x_t = noise_scheduler.add_noise(clean_imgs, noise, timesteps)

            # predizione e ricostruzione one-step
            out = model(x_t, timesteps).sample
            alpha_t = noise_scheduler.alphas_cumprod[timesteps].view(-1,1,1,1)
            sqrt_alpha = torch.sqrt(alpha_t)
            sqrt_1m_alpha = torch.sqrt(1 - alpha_t)
            x0_hat = (x_t - sqrt_1m_alpha * out) / sqrt_alpha
            x0_hat = x0_hat.clamp(-1,1)

            # denormalizza e salva
            clean_np = ((clean_imgs.cpu().numpy() + 1) * 127.5).astype("uint8")
            rec_np   = ((x0_hat.cpu().numpy()   + 1) * 127.5).astype("uint8")

            for i in range(batch_size):
                if saved >= max_examples:
                    return
                img_clean = clean_np[i, 0]
                img_rec   = rec_np[i, 0]
                concat = Image.fromarray(
                    np.concatenate([img_clean, img_rec], axis=1)
                )
                path = os.path.join(output_dir, f"recon_{saved}.png")
                concat.save(path)
                saved += 1

    print(f"Saved {saved} validation reconstructions to {output_dir}")

def save_checkpoint(model, optimizer, epoch, path):
    """
    Salva il checkpoint del modello e dell'ottimizzatore.

    Args:
        model (torch.nn.Module): Il modello da salvare.
        optimizer (torch.optim.Optimizer): L'ottimizzatore da salvare.
        epoch (int): Epoca corrente.
        path (str): Percorso completo per il file `.pth`.
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint salvato in: {path}")
