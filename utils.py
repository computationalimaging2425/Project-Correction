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
                # {"type": "noise_gaussian", "param": {"mean": 0.0, "std": 10.0}},
                # {"type": "noise_salt_pepper", "param": {"prob": 0.02}},
                # {"type": "bright", "param": {"factor": 1.2}},
                # {"type": "contrast", "param": {"factor": 1.3}},
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


def sample_images(
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
