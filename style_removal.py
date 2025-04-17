# This is the code for the Style removal part of the Style Diffusion model.
import torch
from PIL import Image
from diffusers import UNet2DModel, DDIMScheduler, DDIMPipeline
from matplotlib import pyplot as plt

sample_image_size = 256

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Pre-Trained Model
style_removal_model = UNet2DModel.from_pretrained("google/ddpm-celebahq-256").to(device)

# Read the image using PIL
image_path = r"D:\CV_Projects\StyleDiffusion\test2.png"
input_image = Image.open(image_path).convert("RGB")  # Keep RGB channels
plt.imshow(input_image)
plt.show()

from torchvision import transforms
preprocess = transforms.Compose(
    [
        transforms.Resize((sample_image_size, sample_image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Normalize RGB channels
    ]
)

# Diffusion based style removal
K_r = 5 # Iterations to dispel the style
S_for = 8 # Forward steps
S_rev = 50 # Reverse steps
noise_scale = 0.5  # Control strength of style removal

noisy_image = preprocess(input_image).unsqueeze(0).to(device)

noise_scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256")
noise = torch.randn(noisy_image.shape, device=device)

pipe = DDIMPipeline(unet=style_removal_model, scheduler=noise_scheduler)

with torch.no_grad():
    for k in range(K_r):
        print(f"Style removal iteration {k+1}/{K_r}")
        # Forward diffusion
        for i in range(1, S_for+1):
            timestep = torch.LongTensor([i]).to(device)
            noisy_image = noise_scheduler.add_noise(noisy_image, noise * noise_scale, timestep)

        denoised_image = noisy_image.clone()
        noise_scheduler.set_timesteps(num_inference_steps=S_rev)
        # Reverse diffusion
        for i in range(S_rev, 0, -1):
            # print(i)
            timestep = torch.LongTensor([i]).to(device)
            predicted_noise = style_removal_model(denoised_image, timestep).sample
            denoised_image = noise_scheduler.step(predicted_noise, timestep, denoised_image).prev_sample

input_image_content = denoised_image.cpu()
plt.imshow(input_image_content[0].permute(1, 2, 0).numpy())  # Properly display RGB image
plt.show()