import logging
from PIL import Image
from image_scrolling_inpaint import make_image, scroll_image_left, generate_mask_half, generate_mask_full, make_image_inpaint, fix_stitching
from diffusers.utils import load_image
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load image from file

img = load_image("output/00A.PNG")
img.show()

pos=0.5
patch=0.3

mask_width, mask_height = img.size
mask = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)
margin = int(mask_width*patch)
loc = int(mask_width*pos)
mask[:, :loc-margin//2] = 0
mask[:, loc-margin//2:loc+margin//2] = 255
mask[:, loc+margin//2:] = 0
img_mask = Image.fromarray(mask, 'RGB')
img_mask.show()

prompt = "Pixelart background image for a 2D sidescroller videogame, a distant landscape with mountains and a clear sky."

for i in range(0,5):
    #strength = 0.4+(i*0.1)
    strength=0.76
    steps=4+i*2
    print(f"Generating. stregth={strength}, steps={steps}")
    surface = make_image_inpaint(prompt, img, img_mask, mask_width, mask_height, 0.0, steps, strength)
    surface.show()

#surface = make_image_inpaint("", surface, img_mask, mask_width, mask_height, 0.0, 10, 0.45)
#surface = fix_stitching(img, 0.5, 0.25)
surface.show()