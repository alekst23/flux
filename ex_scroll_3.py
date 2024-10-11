import logging
from PIL import Image
from image_utils import make_image, scroll_image_left, generate_mask_half, generate_mask_full, make_image_inpaint, fix_stitching
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def new_img_white(width, height)->Image.Image:
    return Image.new("RGB", (width*3, height), (255, 255, 255))

def new_img_noise(width, height)->Image.Image:
    noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(noise, 'RGB')

def new_img_gradient(width, height)->Image.Image:
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(width):
        value = int((i/width)*255)
        gradient[:, i] = [value, value, value]
    return Image.fromarray(gradient, 'RGB')

def mask_half_margin(mask_width, mask_height)->Image.Image:
    mask = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)
    margin=mask_width//4
    mask[:,:mask_width//2-margin] = 0
    mask[:,mask_width//2-margin:] = 255
    return Image.fromarray(mask, 'RGB')


# Parameters for image generation
width, height = 512, 512
num_inference_steps = 15
guidance_scale = 0.0
strength = 0.775

# Generate and process images based on prompts
prompt = "An alien landscape with futuristic buildings, pixelart background for a 2D videogame."
basedir = "output/07"

logging.info(f"Generating initial image for prompt: '{prompt}'")
img1 = make_image(prompt, width, height, num_inference_steps, guidance_scale, model="dev")
img1.save(f"{basedir}/0.png")
#img1 = load_image("output/001.png")
#img2 = load_image("output/002.png")

# Composition surface
surface = new_img_noise(width*2, height)

# MASK
mask_width, mask_height = surface.size
mask = mask_half_margin(mask_width, mask_height)
#mask.show()

# Prepare first image
surface.paste(img1, (width,0))
#surface.show()

# Composition loop
for i in range(0,10):
    logging.info(f"LOOP {i}")

    # Slide
    logging.info("- slide")
    surface = scroll_image_left(surface)

    # Generate new image
    logging.info("- new image")
    img2 = make_image(prompt, width, height, num_inference_steps//4, guidance_scale, model="dev")
    img2.save(f"{basedir}/1.png")

    # Add to composition
    logging.info("- combine")
    surface.paste(img2, (width,0))
    #surface.show()

    # Inpaint
    logging.info("- inpaint")
    surface = make_image_inpaint(prompt, surface, mask, mask_width, mask_height, guidance_scale, num_inference_steps, strength)
    
    # Fix stitching
    logging.info("- patch")
    surface = fix_stitching(surface, 0.25, 0.15, strength=0.75, steps=15)
    #surface.show()

    # Save to output
    logging.info("- save")
    # Get a cropped left half of the image
    cropped_image = surface.crop((0, 0, width, height))
    cropped_image.save(f"{basedir}/000{i}.png")
    #surface.save(f"{basedir}/000{i}.png")

