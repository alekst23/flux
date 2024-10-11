import logging
from PIL import Image
from image_utils import make_image, scroll_image_left, generate_mask_half, generate_mask_full, make_image_inpaint
from diffusers.utils import load_image
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parameters for image generation
width, height = 1024, 512
num_inference_steps = 4
guidance_scale = 0.0
strength = 0.86

# Generate and process images based on prompts
prompt = "Pixelart background image for a 2D sidescroller videogame, continuous landscape with mountains and a clear sky."

logging.info(f"Generating initial image for prompt: '{prompt}'")
img1=make_image(prompt, width, height, num_inference_steps, guidance_scale)#.save("output/001.png")
img2=make_image(prompt, width, height, num_inference_steps, guidance_scale)#.save("output/002.png")

# img1 = load_image("output/001.png")
# img2 = load_image("output/002.png")

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

img_compose = new_img_noise(width*2, height)

img_compose.paste(img1, (0,0))
img_compose.paste(img2, (width,0))

img_compose.show()

# MASK
mask_width, mask_height = img_compose.size
# mask = Image.new('L', (mask_width, mask_height))
# mask_np = np.array(mask)
# # middle section of mask is white
# # Calculate the segment widths
# segment_width = mask_width // 3
# segment_margin = segment_width // 3
# # Create the mask with three equal segments
# mask_np[:, :segment_width-segment_margin] = 0  # Black segment
# mask_np[:, segment_width-segment_margin:2*segment_width+segment_margin] = 255  # White segment
# mask_np[:, 2*segment_width+segment_margin:] = 0  # Black segment
# mask = Image.fromarray(mask_np)
mask = mask_half_margin(mask_width, mask_height)
mask.show()

# img_new = make_image_inpaint(prompt, img_compose, mask, width*3, height, guidance_scale, num_inference_steps, strength)
# img_new.show()
for i in range(0,5):
    strength=0.75+(0.01*i)
    logging.info(f"Running generation. stregth={strength}")
    make_image_inpaint(prompt, img_compose, mask, mask_width, mask_height, guidance_scale, 4, strength).show()

strength=0.80
for i in range(0,5):
    steps = 2+2*i
    logging.info(f"Running generation. steps={steps}")
    make_image_inpaint(prompt, img_compose, mask, mask_width, mask_height, guidance_scale, steps, strength).show()