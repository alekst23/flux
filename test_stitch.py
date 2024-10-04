import logging
from PIL import Image
from image_scrolling_inpaint import make_image, scroll_image_left, generate_mask_half, generate_mask_full, make_image_inpaint
from diffusers.utils import load_image
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load image from file

surface = load_image("output/03/0006.png")
#surface.show()

mask_width, mask_height = surface.size
mask = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)
margin=mask_width//4
mask[:, :margin//2] = 0
mask[:, margin//2:margin//2+margin] = 255
mask[:, margin//2+margin:] = 0
img_mask = Image.fromarray(mask, 'RGB')
#img_mask.show()

prompt = "Pixelart background image for a 2D sidescroller videogame, a distant landscape with mountains and a clear sky."

# for i in range(0,4):
#     strength = 0.3+(i*0.1)
#     print(f"Generating. stregth={strength}")
#     surface = make_image_inpaint(prompt, surface, img_mask, mask_width, mask_height, 0.0, 10, strength)
#     surface.show()

surface = make_image_inpaint("", surface, img_mask, mask_width, mask_height, 0.0, 10, 0.45)
surface.show()