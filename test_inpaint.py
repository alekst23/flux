from diffusers import FluxInpaintPipeline
import torch
import matplotlib.pyplot as plt
from diffusers.utils import load_image

ckpt_id = "black-forest-labs/FLUX.1-schnell"

def make_image_inpaint(prompt, input_image, mask_image, width, height, guidance_scale=0.3, num_inference_steps=4):
    pipe = FluxInpaintPipeline.from_pretrained(
        ckpt_id,
        torch_dtype=torch.bfloat16
    )
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.enable_sequential_cpu_offload()

    image = pipe(
        prompt=prompt,
        image=input_image,
        mask_image=mask_image,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images[0]

    return image

# Example usage of the modified function
prompt="Face of a yellow cat, high resolution, sitting on a park bench"
height = 512
width = 512

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
source = load_image(img_url)
mask = load_image(mask_url)

image = make_image_inpaint(prompt, source, mask, width, height)
plt.imshow(image)
plt.show()
