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
IMAGE_SIZE = 128
BATCH_SIZE = 16


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
    "noise_gaussian": lambda img, param, size: add_gaussian_noise(img, **param).resize(
        (size, size), Image.BICUBIC
    ),
    "noise_salt_pepper": lambda img, param, size: add_salt_pepper(img, **param).resize(
        (size, size), Image.BICUBIC
    ),
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
                {"type": None},  # no augmentation
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
    epoch=0,
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
            alpha_t = noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
            sqrt_alpha = torch.sqrt(alpha_t)
            sqrt_1m_alpha = torch.sqrt(1 - alpha_t)
            x0_hat = (x_t - sqrt_1m_alpha * out) / sqrt_alpha
            x0_hat = x0_hat.clamp(-1, 1)

            # denormalizza e salva
            clean_np = ((clean_imgs.cpu().numpy() + 1) * 127.5).astype(np.uint8)
            rec_np = ((x0_hat.cpu().numpy() + 1) * 127.5).astype(np.uint8)

            for i in range(batch_size):
                if saved >= max_examples:
                    return
                img_clean = clean_np[i, 0]
                img_rec = rec_np[i, 0]
                concat = Image.fromarray(np.concatenate([img_clean, img_rec], axis=1))

                path = os.path.join(output_dir, f"recon_{epoch}_{saved}.png")
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


def load_checkpoint(
    model_save_dir,
    model_to_load_name,
    model,
    optimizer=None,
    device=torch.device("cpu"),
):
    import os
    import torch

    loaded = False
    ckpt_path = os.path.join(model_save_dir, model_to_load_name)

    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}, starting fresh on {device}.")
        return loaded, model, optimizer, 0

    checkpoint = torch.load(ckpt_path, map_location=device)

    # Rimuovi il prefisso "_orig_mod." dalle chiavi se presente
    new_state_dict = {}
    for k, v in checkpoint["model"].items():
        if k.startswith("_orig_mod."):
            new_k = k.replace("_orig_mod.", "")
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    loaded = True
    start_epoch = checkpoint.get("epoch", 0)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Loaded optimizer state from checkpoint '{ckpt_path}'")
    else:
        print("Optimizer state not found in checkpoint, starting with a new optimizer.")

    print(f"Loaded checkpoint '{ckpt_path}' (epoch {start_epoch})")

    model.eval()

    if loaded:
        print(
            f"Model {model_to_load_name} loaded successfully to {device}, starting from epoch {start_epoch}."
        )
    else:
        print(f"Model {model_to_load_name} not found, starting fresh on {device}.")

    return loaded, model, optimizer, start_epoch


LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5


def get_unet_model(
    sample_size=128,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(64, 128, 256),
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    dropout=0.1,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    device="cpu",
):
    """
    Function to create a UNet model for image generation.

    Returns:
    UNet2DModel: A UNet model configured for image generation.
    """

    from diffusers import UNet2DModel

    # Create the UNet model with the specified parameters
    model = UNet2DModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=layers_per_block,
        block_out_channels=block_out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        dropout=dropout,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Move the model to the specified device
    model.to(device)
    print(f"Model moved to {device}")

    return model, optimizer


def setup_environment(on_colab=False):
    """
    returns:
        - data_root: path to the raw data directory
        - model_save_dir: path to the model save directory
        - train_dir: path to the training data directory
        - test_dir: path to the testing data directory
    """
    if on_colab:
        from google.colab import drive

        drive.mount(os.getenv("GOOGLE_DRIVE_CONTENT_PATH", "/content/drive"))
        data_root = os.getenv(
            "GOOGLE_DRIVE_RAW_DATA_DIR",
            "/content/drive/MyDrive/ComputationalImaging/raw_data",
        )
        model_save_dir = os.getenv(
            "GOOGLE_DRIVE_MODEL_SAVE_DIR",
            "/content/drive/MyDrive/ComputationalImaging/checkpoints",
        )
    else:
        data_root = os.getenv("RAW_DATA_DIR", "raw_data")
        model_save_dir = os.getenv("MODEL_SAVE_DIR", "checkpoints")

    os.makedirs(model_save_dir, exist_ok=True)
    print(f"Data root: {data_root}")
    print(f"Model save directory: {model_save_dir}")

    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")

    return data_root, model_save_dir, train_dir, test_dir


def print_model_summary(model):
    print("Model Summary:")
    print(f"Model Type: {type(model).__name__}")
    print(f"Number of Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Device: {next(model.parameters()).device}")


def get_dataloader(
    root_dir,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
):
    dataset = AugmentedDataset(root_dir, image_size=image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataloader


def get_schedulers():
    from diffusers import DDPMScheduler, DDIMScheduler

    noise_scheduler = DDPMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)
    ddim_scheduler = DDIMScheduler(
        beta_start=noise_scheduler.config.beta_start,
        beta_end=noise_scheduler.config.beta_end,
        beta_schedule=noise_scheduler.config.beta_schedule,
        clip_sample=True,
    )
    ddim_scheduler.set_timesteps(NUM_TRAIN_TIMESTEPS)

    return noise_scheduler, ddim_scheduler
