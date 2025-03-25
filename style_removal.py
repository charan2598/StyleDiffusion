# This is the code for the Style removal part of the Style Diffusion model.
import torch
from PIL import Image
from diffusers import UNet2DModel, DDIMScheduler, DDIMPipeline

sample_image_size = 128

# Pre-Trained Model
style_removal_model = UNet2DModel(
    sample_size=sample_image_size,  # the target image resolution
    in_channels=1,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

# Read the image using PIL and convert it to grayscale.
image_path = r"D:\CV_Projects\StyleDiffusion\starynight.jpeg"
input_image = Image.open(image_path).convert('L')

from torchvision import transforms
preprocess = transforms.Compose(
    [
        transforms.Resize((sample_image_size, sample_image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

# Diffusion based style removal
K_r = 5 # Iterations to dispel the style
S_for = 40 # Forward steps
S_rev = 40 # Reverse steps

noisy_image = preprocess(input_image).unsqueeze(0) # We might want to scale to [-1, 1]

noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
noise = torch.randn(noisy_image.shape)

pipe = DDIMPipeline(unet=style_removal_model, scheduler=noise_scheduler)

with torch.no_grad():
    for i in range(K_r):
        # Forward diffusion
        
        for i in range(1, S_for+1):
            timestep = torch.LongTensor([i])
            noisy_image = noise_scheduler.add_noise(noisy_image, noise, timestep)

        denoised_image = noisy_image.clone()
        noise_scheduler.set_timesteps(num_inference_steps=S_rev)
        # Reverse diffusion
        for i in range(S_rev, 1, -1):
            timestep = torch.LongTensor([i])
            predicted_noise = style_removal_model(denoised_image, timestep).sample
            denoised_image = noise_scheduler.step(predicted_noise, timestep, denoised_image).prev_sample

input_image_content = denoised_image
