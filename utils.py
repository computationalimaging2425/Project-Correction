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

def add_gaussian_noise(img: Image.Image, mean=0., std=10.):
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


class AugmentedDataset(Dataset):
    def __init__(self, root_dir, image_size=128):
        # dataset base che carica le immagini in grayscale e le ridimensiona
        self.base_ds = datasets.ImageFolder(
            root=root_dir,
            transform=transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
            ])
        )
        self.angles = [-5, -3, 3, 5]
        self.noise_types = ["gaussian", "salt_pepper"]
        # lista di tutte le possibili trasformazioni (None = immagine originale)
        self.augs = [None]  \
            + [("rot", a) for a in self.angles]  \
            + [("flip", None)]  \
            + [("noise", "gaussian"), ("noise", "salt_pepper")]  \
            + [("bright", None), ("contrast", None)]
        self.image_size = image_size

        # fine pipeline per convertire PIL→Tensor normalizzato [-1,1]
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ])

    def __len__(self):
        return len(self.base_ds) * len(self.augs)

    def __getitem__(self, idx):
        # individuo immagine base + tipo di augment
        img_idx = idx // len(self.augs)
        aug_idx = idx % len(self.augs)

        img, label = self.base_ds[img_idx]      # PIL in grayscale 128×128
        aug = self.augs[aug_idx]

        # applica la augmentazione scelta
        if aug is not None:
            mode, param = aug
            if mode == "rot":
                img = rotate_fixed(img, param, self.image_size)
            elif mode == "flip":
                img = horizontal_flip(img)
            elif mode == "noise":
                if param == "gaussian":
                    img = add_gaussian_noise(img, mean=0.0, std=10.0)
                else:
                    img = add_salt_pepper(img, prob=0.02)
                img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
            elif mode == "bright":
                img = change_brightness(img, factor=1.2)
                img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
            elif mode == "contrast":
                img = change_contrast(img, factor=1.3)
                img = img.resize((self.image_size, self.image_size), Image.BICUBIC)

        # infine trasformo in tensor e normalizzo
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
