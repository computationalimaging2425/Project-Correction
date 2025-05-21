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
    save_images=False,
):
    """
    Per ogni batch in test_loader:
      • prende un'immagine pulita x0
      • aggiunge rumore x_t = q(x_t|x0) con t casuale
      • predice l'epsilon con la U-Net
      • ricostruisce x0_hat da x_t e epsilon_pred
      • ritorna due liste: immagini originali e ricostruite (primi max_examples)
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    to_pil = transforms.ToPILImage()

    orig_images = []
    rec_images = []
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

            # denormalizza
            clean_np = ((clean_imgs.cpu().numpy() + 1) * 127.5).astype(np.uint8)
            rec_np = ((x0_hat.cpu().numpy() + 1) * 127.5).astype(np.uint8)

            for i in range(batch_size):
                if saved >= max_examples:
                    # ritorna liste quando raggiunto max
                    return orig_images, rec_images

                # estrai e converte
                img_clean_array = clean_np[i, 0]
                img_rec_array = rec_np[i, 0]
                img_clean = Image.fromarray(img_clean_array)
                img_rec = Image.fromarray(img_rec_array)

                # salva concatenazione su disco
                concat = Image.fromarray(
                    np.concatenate([img_clean_array, img_rec_array], axis=1)
                )
                path = os.path.join(
                    output_dir, f"reconstructed_epoch_{epoch}_{saved}.png"
                )
                if save_images:
                    concat.save(path)

                # aggiunge liste
                orig_images.append(img_clean)
                rec_images.append(img_rec)
                saved += 1

    print(f"Saved {saved} validation reconstructions to {output_dir}")
    return orig_images, rec_images


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


@torch.no_grad()
def dps_deblur(
    model,
    ddim_scheduler,
    y,  # immagine sfocata, tensor (B, C, H, W)
    K,  # operatore di blur
    noise_scheduler,
    eta=0.0,  # controlla la stochasticità (0=deterministico)
    sigma_y=0.01,  # varianza del rumore di misura
    device="cpu",
):
    """
    Applica DPS per il deblurring usando il modello di diffusione UNet.

    Args:
        model: modello UNet pre-addestrato
        scheduler: DDIM scheduler per diffusione inversa
        y: immagine sfocata (batch_size, channels, H, W)
        K: operatore blur (funzione che applica blur e il suo adjoint)
        noise_scheduler: scheduler del rumore (usato per ottenere sigma_t)
        num_inference_steps: numero di passi di campionamento
        eta: param. per la diffusione DDIM
        device: device CUDA o CPU

    Returns:
        x_recon: immagine ricostruita (B, C, H, W)
    """
    import torch
    import torch.nn.functional as F
    from tqdm.auto import tqdm

    B = y.shape[0]
    # timesteps from scheduler (if custom num steps given override)
    timesteps = ddim_scheduler.timesteps

    # initialize with noise
    x_t = torch.randn_like(y).to(device)
    y = y.to(device)

    for t in tqdm(timesteps, desc="DPS Deblurring", unit="step", leave=False):
        t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
        # 1) UNet predicts noise
        out = model(x_t, t_tensor)
        eps_theta = out.sample

        # 2) compute x0 pred
        alpha = noise_scheduler.alphas_cumprod[t].to(device)
        sqrt_alpha = torch.sqrt(alpha)
        sqrt_1m_alpha = torch.sqrt(1 - alpha)
        x0_pred = (x_t - sqrt_1m_alpha * eps_theta) / sqrt_alpha

        # 3) DPS posterior update: gradient step towards data
        # gradient of ||y - K x||^2 wrt x at x0_pred: -K^T(y - K x0_pred)
        # step size gamma_t = sigma_prior^2 / (sigma_y^2 + sigma_prior^2)
        sigma_prior2 = (1 - alpha).item()
        gamma = sigma_prior2 / (sigma_y**2 + sigma_prior2)
        # apply update
        res = y - K(x0_pred)
        x0_post = x0_pred + gamma * K.T(res)

        # 4) DDIM update to next timestep
        # estimate noise using posterior x0
        eps_post = (x_t - sqrt_alpha * x0_post) / sqrt_1m_alpha
        # parameters for previous step
        prev_t = max(t - ddim_scheduler.config.num_train_timesteps // len(timesteps), 0)
        alpha_prev = noise_scheduler.alphas_cumprod[prev_t].to(device)
        sqrt_alpha_prev = torch.sqrt(alpha_prev)
        sqrt_1m_alpha_prev = torch.sqrt(1 - alpha_prev)

        # compute sigma for ddim
        sigma = eta * torch.sqrt(
            (1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)
        )

        # update x_{t-1}
        x_t = (
            sqrt_alpha_prev * x0_post
            + sqrt_1m_alpha_prev * eps_post
            + sigma * torch.randn_like(x_t)
        )

    return x_t


def dps_deblur_and_plot(
    num_images=5,
    test_loader=None,
    K=None,
    model=None,
    device="cpu",
    ddim_scheduler=None,
    noise_scheduler=None,
):
    """
    Picks `num_images` random images from the test set,
    applies motion blur, reconstructs via DPS, and plots results.
    """
    import random
    import matplotlib.pyplot as plt
    from IPPy import metrics
    from tqdm.auto import tqdm

    # Random selection of indices
    indices = random.sample(range(len(test_loader.dataset)), k=num_images)

    # Prepare figure with num_images rows and 3 columns
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))

    for row, idx in enumerate(tqdm(indices, desc="Processing images", unit="image")):
        # Load, blur, reconstruct
        x_true = test_loader.dataset[idx][0].unsqueeze(0).to(device)
        y = K(x_true)
        x_recon = dps_deblur(
            model=model,
            ddim_scheduler=ddim_scheduler,
            y=y,
            K=K,
            noise_scheduler=noise_scheduler,
            eta=0.0,
            device=device,
        )

        # Compute metrics
        psnr_deg = metrics.PSNR(x_true.cpu(), y.cpu())
        ssim_deg = metrics.SSIM(x_true.cpu(), y.cpu())
        psnr_rec = metrics.PSNR(x_true.cpu(), x_recon.cpu())
        ssim_rec = metrics.SSIM(x_true.cpu(), x_recon.cpu())

        # Titles with bold for higher
        psnr_blur_str = f"PSNR: {psnr_deg:.4f} dB"
        psnr_recon_str = f"PSNR: {psnr_rec:.4f} dB"
        ssim_blur_str = f"SSIM: {ssim_deg:.4f}"
        ssim_recon_str = f"SSIM: {ssim_rec:.4f}"

        if psnr_deg > psnr_rec:
            psnr_blur_str = f"$\\bf{{{psnr_blur_str}}}$"
        else:
            psnr_recon_str = f"$\\bf{{{psnr_recon_str}}}$"
        if ssim_deg > ssim_rec:
            ssim_blur_str = f"$\\bf{{{ssim_blur_str}}}$"
        else:
            ssim_recon_str = f"$\\bf{{{ssim_recon_str}}}$"

        # Plot original
        ax_orig = axes[row, 0] if num_images > 1 else axes[0]
        ax_orig.imshow(x_true.cpu().numpy()[0, 0], cmap="gray")
        ax_orig.axis("off")
        if row == 0:
            ax_orig.set_title("Original")

        # Plot blurred
        ax_blur = axes[row, 1] if num_images > 1 else axes[1]
        ax_blur.imshow(y.cpu().numpy()[0, 0], cmap="gray")
        ax_blur.axis("off")
        if row == 0:
            ax_blur.set_title(f"Blurred\n{psnr_blur_str}\n{ssim_blur_str}")
        else:
            # Only metrics in subsequent rows
            ax_blur.set_title(f"{psnr_blur_str}\n{ssim_blur_str}")

        # Plot reconstructed
        ax_recon = axes[row, 2] if num_images > 1 else axes[2]
        ax_recon.imshow(x_recon.cpu().numpy()[0, 0], cmap="gray")
        ax_recon.axis("off")
        if row == 0:
            ax_recon.set_title(f"Reconstructed\n{psnr_recon_str}\n{ssim_recon_str}")
        else:
            ax_recon.set_title(f"{psnr_recon_str}\n{ssim_recon_str}")

    fig.suptitle("Deblurring con DPS", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def red_diff(
    model,
    ddim_scheduler,
    y,  # degraded measurement tensor (B, C, H, W)
    K,  # measurement operator with forward K(x) and adjoint K.T(y)
    noise_scheduler,
    sigma_y=0.01,  # assumed measurement noise std
    lambda_scale=0.25,  # base weight for regularization
    lr=0.1,
    reg_strategy="linear",  # one of: linear, sqrt, square, log, clip, const
    device="cpu",
):
    """
    RED-Diff optimization-based sampler for inverse problems.

    Args:
        model: diffusion model predicting noise (eps_theta)
        ddim_scheduler: object containing ordered timesteps
        y: observed measurement (B, C, H, W)
        K: degradation operator with K(x) and K.T(y)
        noise_scheduler: diffusion noise scheduler with alphas_cumprod
        sigma_y: std of measurement noise
        lambda_scale: regularization strength
        lr: learning rate
        reg_strategy: weighting scheme for denoising loss
        device: 'cpu' or 'cuda'

    Returns:
        mu: reconstructed image
    """
    import torch
    import torch.nn.functional as F
    from tqdm.auto import tqdm

    # Initialize reconstruction from adjoint
    mu = K.T(y).detach().to(device)
    mu.requires_grad_(True)
    optimizer = torch.optim.Adam([mu], lr=lr)

    T = noise_scheduler.alphas_cumprod.shape[0] - 1
    timesteps = ddim_scheduler.timesteps

    for t in tqdm(timesteps, desc="RED-Diff sampling", unit="step", leave=False):
        t_batch = torch.full((mu.shape[0],), t, dtype=torch.long, device=device)

        alpha_cumprod = noise_scheduler.alphas_cumprod[t].to(device)
        sqrt_alpha = torch.sqrt(alpha_cumprod)
        sigma_t = torch.sqrt(1 - alpha_cumprod)

        # Sample noise and perform forward diffusion
        eps = torch.randn_like(mu)
        x_t = sqrt_alpha * mu + sigma_t * eps

        # Predict noise (eps_theta) from the model
        model_out = model(x_t, t_batch)
        eps_theta = model_out.sample  # or model_out if not using .sample

        # Data fidelity term (measurement consistency)
        loss_obs = F.mse_loss(K(mu), y) / (2 * sigma_y**2)

        # Compute inverse SNR for weight scaling
        snr_inv = sigma_t / sqrt_alpha  # equivalent to 1/SNR

        # Apply weighting strategy
        if reg_strategy == "sqrt":
            w_t = snr_inv.sqrt()
        elif reg_strategy == "square":
            w_t = snr_inv**2
        elif reg_strategy == "log":
            w_t = torch.log1p(snr_inv)
        elif reg_strategy == "clip":
            w_t = torch.clamp(snr_inv, max=1.0)
        elif reg_strategy == "const":
            w_t = torch.ones_like(snr_inv)
        else:  # default: linear
            w_t = snr_inv

        # Regularization term (noise guidance)
        loss_reg = (w_t.view(-1, 1, 1, 1) * (eps_theta.detach() - eps).pow(2)).mean()
        loss = loss_obs + lambda_scale * loss_reg

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return mu.detach()


def red_diff_and_plot(
    num_images=5,
    test_loader=None,
    K=None,
    model=None,
    device="cpu",
    noise_scheduler=None,
    ddim_scheduler=None,
):
    """
    Applies RED-Diff to random images from test_loader, then plots original, degraded, and reconstructed.

    Args:
        num_images: number of images to visualize
        test_loader: DataLoader for dataset
        K: degradation operator
        model: diffusion UNet model
        noise_scheduler: noise scheduler
        device: computation device
    """
    import matplotlib.pyplot as plt
    from IPPy import metrics
    import random
    from tqdm.auto import tqdm

    # Select random indices
    indices = random.sample(range(len(test_loader.dataset)), num_images)

    # Prepare figure with num_images rows and 3 columns
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))

    for row, idx in enumerate(tqdm(indices, desc="Processing images", unit="image")):
        # Load, blur, reconstruct
        x_true = test_loader.dataset[idx][0].unsqueeze(0).to(device)
        y = K(x_true)
        x_recon = red_diff(
            model=model,
            noise_scheduler=noise_scheduler,
            y=y,
            K=K,
            ddim_scheduler=ddim_scheduler,
            device=device,
        )

        # Compute metrics
        psnr_deg = metrics.PSNR(x_true.cpu(), y.cpu())
        ssim_deg = metrics.SSIM(x_true.cpu(), y.cpu())
        psnr_rec = metrics.PSNR(x_true.cpu(), x_recon.cpu())
        ssim_rec = metrics.SSIM(x_true.cpu(), x_recon.cpu())

        # Titles with bold for higher
        psnr_blur_str = f"PSNR: {psnr_deg:.4f} dB"
        psnr_recon_str = f"PSNR: {psnr_rec:.4f} dB"
        ssim_blur_str = f"SSIM: {ssim_deg:.4f}"
        ssim_recon_str = f"SSIM: {ssim_rec:.4f}"

        if psnr_deg > psnr_rec:
            psnr_blur_str = f"$\\bf{{{psnr_blur_str}}}$"
        else:
            psnr_recon_str = f"$\\bf{{{psnr_recon_str}}}$"
        if ssim_deg > ssim_rec:
            ssim_blur_str = f"$\\bf{{{ssim_blur_str}}}$"
        else:
            ssim_recon_str = f"$\\bf{{{ssim_recon_str}}}$"

        # Plot original
        ax_orig = axes[row, 0] if num_images > 1 else axes[0]
        ax_orig.imshow(x_true.cpu().numpy()[0, 0], cmap="gray")
        ax_orig.axis("off")
        if row == 0:
            ax_orig.set_title("Original")

        # Plot blurred
        ax_blur = axes[row, 1] if num_images > 1 else axes[1]
        ax_blur.imshow(y.cpu().numpy()[0, 0], cmap="gray")
        ax_blur.axis("off")
        if row == 0:
            ax_blur.set_title(f"Blurred\n{psnr_blur_str}\n{ssim_blur_str}")
        else:
            # Only metrics in subsequent rows
            ax_blur.set_title(f"{psnr_blur_str}\n{ssim_blur_str}")

        # Plot reconstructed
        ax_recon = axes[row, 2] if num_images > 1 else axes[2]
        ax_recon.imshow(x_recon.cpu().numpy()[0, 0], cmap="gray")
        ax_recon.axis("off")
        if row == 0:
            ax_recon.set_title(f"Reconstructed\n{psnr_recon_str}\n{ssim_recon_str}")
        else:
            ax_recon.set_title(f"{psnr_recon_str}\n{ssim_recon_str}")

    fig.suptitle("RED-Diff Reconstruction", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
