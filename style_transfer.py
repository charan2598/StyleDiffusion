import torch
from PIL import Image
from diffusers import DDIMPipeline, DDIMScheduler, UNet2DModel
from torchvision import transforms

sample_image_size = 128

# Read the image using PIL and convert it to grayscale.
image_path = r"D:\CV_Projects\StyleDiffusion\starynight.jpeg"
content_image = Image.open(image_path).convert('L')
style_image = Image.open(image_path).convert('L')

preprocess = transforms.Compose(
    [
        transforms.Resize((sample_image_size, sample_image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

# Pre-Trained Model
style_transfer_model = UNet2DModel(
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


# TODO: We basically need to run this over a 
# bunch of content images and style images
# for now running just on 1 of each.
content_image = preprocess(content_image).unsqueeze(0)
style_image = preprocess(style_image).unsqueeze(0)

noisy_image = torch.cat([content_image, style_image], dim=0)

noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
noise = torch.randn(noisy_image.shape)

# Diffusion based style transfer
K = 5 # Iterations
K_s = 50 # iterations for reconstruction of style
S_for = 40 # Forward steps
S_rev = 6 # Reverse steps
    
for i in range(1, S_for+1):
    timestep = torch.LongTensor([i])
    noisy_image = noise_scheduler.add_noise(noisy_image, noise, timestep)

noisy_content_image = noisy_image[:1,...]
noisy_style_image = noisy_image[1:,...]

# Fine tune the diffusion model
for i in range(K):
    # Optimize reconstruction loss
    for j in range(K_s):
        denoised_style_image = noisy_style_image.clone()
        noise_scheduler.set_timesteps(num_inference_steps=S_rev)
        # Reverse diffusion
        for i in range(S_rev, 1, -1):
            timestep = torch.LongTensor([i])
            predicted_noise = style_transfer_model(denoised_style_image, timestep).sample
            denoised_style_image = noise_scheduler.step(predicted_noise, timestep, denoised_style_image).prev_sample
            # Compute style reconstruction loss here and backward call

    # Optimize the style disentanglement loss
    for i in range(1): # Ideally iterate over the sample
        denoised_content_image = noisy_content_image.clone()
        noise_scheduler.set_timesteps(num_inference_steps=S_rev)
        # Reverse diffusion
        for i in range(S_rev, 1, -1):
            timestep = torch.LongTensor([i])
            predicted_noise = style_transfer_model(denoised_content_image, timestep).sample
            denoised_content_image = noise_scheduler.step(predicted_noise, timestep, denoised_content_image).prev_sample
            # compute style disentanglement loss and backward
            