import os
import torch
from tqdm import tqdm
from torchvision import transforms

NUM_TRAIN_TIMESTEPS = 1000

def sample_images(output_path="result/ddim_sample.png", num_steps=NUM_TRAIN_TIMESTEPS, DEVICE="cpu", IMAGE_SIZE=128, model=None, ddim_scheduler=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    with torch.no_grad():
        sample = torch.randn((1, 1, IMAGE_SIZE, IMAGE_SIZE), device=DEVICE)

        for t in tqdm(reversed(range(0, num_steps)), desc="Sampling DDIM", total=num_steps):
            noise_pred = model(sample, torch.tensor([t], device=DEVICE)).sample
            sample = ddim_scheduler.step(noise_pred, t, sample).prev_sample

        # denormalize and save
        final = (sample.clamp(-1, 1) + 1) / 2
        transforms.ToPILImage()(final.squeeze(0).squeeze(0).cpu()).save(output_path)
        print(f"Sample saved to {output_path}")