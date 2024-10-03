from diffusers import FluxPipeline, FluxInpaintPipeline
import torch
from diffusers.utils import load_image
import numpy as np
from PIL import Image

ckpt_id = "black-forest-labs/FLUX.1-schnell"

def make_image_inpaint(prompt, input_image, mask_image, width, height, guidance_scale=0.3, num_inference_steps=4):
    """
    Generates an image using the FluxInpaintPipeline based on a prompt, input image, and mask image.
    
    :param prompt: Text prompt for the inpainting model.
    :param input_image: The input image to inpaint.
    :param mask_image: The mask image to define the inpainting area.
    :param width: The width of the output image.
    :param height: The height of the output image.
    :param guidance_scale: The guidance scale for the inpainting model.
    :param num_inference_steps: The number of inference steps for the inpainting model.
    :return: The inpainted image.
    """
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

def scroll_image_left(image, shift_fraction=0.5):
    """
    Scrolls the image to the left by a specified fraction of its width.
    
    :param image: The input image to scroll.
    :param shift_fraction: The fraction of the width to scroll the image by.
    :return: The scrolled image with the right half filled with transparent pixels.
    """
    width, height = image.size
    shift_amount = int(width * shift_fraction)
    
    # Create a new image with the same dimensions and a transparent background
    new_image = Image.new("RGB", (width, height), (0, 0, 0))
    
    # Paste the shifted part of the original image into the new image
    new_image.paste(image.crop((shift_amount, 0, width, height)), (0, 0))
    
    return new_image

def generate_mask(image):
    """
    Generates a mask that's black on the left half and white on the right half.
    
    :param image: The input image to base the mask on.
    :return: The generated mask image.
    """
    width, height = image.size
    mask = Image.new('L', (width, height))
    mask_np = np.array(mask)
    mask_np[:, :width // 2] = 0  # Black on the left half
    mask_np[:, width // 2:] = 255  # White on the right half
    return Image.fromarray(mask_np)

def inpaint_image(prompt, input_image_path):
    """
    Inpaints an image based on a prompt and an input image file path.
    
    :param prompt: Text prompt for the inpainting model.
    :param input_image_path: The file path of the input image.
    :return: The inpainted image.
    """
    input_image = load_image(input_image_path)
    mask_image = generate_mask(input_image)
    inpainted_image = make_image_inpaint(prompt, input_image, mask_image, input_image.width, input_image.height)
    return inpainted_image

def combine_image_and_mask(image, mask):
    """
    Combines the original image and the mask to create a new inpainted image.
    
    :param image: The original input image.
    :param mask: The mask image used for inpainting.
    :return: The combined inpainted image.
    """
    return make_image_inpaint("Inpaint this image", image, mask, image.width, image.height)

def make_image(prompt, width, height, num_inference_steps=10, guidance_scale=0.3):
    """
    Generates an image using the FluxPipeline based on a prompt.
    
    :param prompt: Text prompt for the image generation model.
    :param width: The width of the output image.
    :param height: The height of the output image.
    :param num_inference_steps: The number of inference steps for the image generation model.
    :param guidance_scale: The guidance scale for the image generation model.
    :return: The generated image.
    """
    pipe = FluxPipeline.from_pretrained(
        ckpt_id,
        torch_dtype=torch.bfloat16,
    )
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.enable_sequential_cpu_offload()

    image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        max_sequence_length=512
    ).images[0]

    return image
