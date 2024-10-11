from diffusers import FluxInpaintPipeline
import torch
import matplotlib.pyplot as plt
from diffusers.utils import load_image
import logging
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ckpt_id = "black-forest-labs/FLUX.1-schnell"

def print_image_properties(image, image_name):
    logging.info(f"Properties of {image_name}:")
    logging.info(f"Mode: {image.mode}")
    logging.info(f"Size: {image.size}")
    logging.info(f"Format: {image.format}")
    logging.info(f"Info: {image.info}")
    if hasattr(image, 'palette') and image.palette:
        logging.info(f"Palette: {image.palette.getdata()}")
    if "icc_profile" in image.info:
        logging.info(f"ICC Profile: Present")
    else:
        logging.info(f"ICC Profile: Not present")


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
prompt="Face of a green dragon, high resolution, sitting on a park bench"
height = 512
width = 512

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
source = load_image(img_url)
mask = load_image(mask_url)

print_image_properties(mask, "Mask Image")

image = make_image_inpaint(prompt, source, mask, width, height)
plt.imshow(image)
plt.show()
