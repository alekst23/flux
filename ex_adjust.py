import logging
from PIL import Image, ImageDraw
from image_utils import make_image, make_image_inpaint, new_img_white, new_img_noise
from diffusers.utils import load_image
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def template_horizon(width, height, pos: float=0.6)->Image.Image:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[int(height*pos):, :] = 50
    canvas[:int(height*pos), :] = 255
    return Image.fromarray(canvas, 'RGB')

def template_noisy_horizon(width, height, pos: float=0.6)->Image.Image:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[int(height*pos):, :] = 50
    canvas[:int(height*pos), :] = 255
    noise = np.random.randint(50, 100, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(canvas+noise, 'RGB')

def mask_center_square(width, height, size=100)->Image.Image:
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    mask[height//2-size//2:height//2+size//2, width//2-size//2:width//2+size//2] = 255
    return Image.fromarray(mask, 'RGB')

width, height = 1024, 512

img = load_image ("output/00E.PNG")

mask = mask_center_square(width, height, size=300)
#mask.show()

prompt = "The door is open inwards revealing a long hallway lit by torches."
scale=0.0
strength=0.78
steps=15
# surface = make_image_inpaint(prompt, img, mask, width, height, num_inference_steps=steps, strength=strength, guidance_scale=scale, model="dev")
# surface.show()

for i in range(0, 9):
    scale = 0.0 + 0.1*i
    strength = 0.7
    steps = 12
    print_stats=f"steps={steps}, scale={scale}, strength={strength}"
    logging.info(f"GENERATING, {print_stats}")
    logging.info("-dev")
    #surface = make_image(prompt, width, height, num_inference_steps=steps, guidance_scale=scale, model="dev")
    surface = make_image_inpaint(prompt, img, mask, width, height, num_inference_steps=steps, strength=strength, guidance_scale=scale, model="dev")
    #surface.show()
    # add text to surface
    ImageDraw.Draw(surface).text((10, 10), print_stats, (200, 255, 200))
    surface.show()

#     logging.info("-schnell")
#     #surface = make_image(prompt, width, height, num_inference_steps=steps//2, guidance_scale=scale, model="schnell")
#     surface = make_image_inpaint(prompt, img, mask, width, height, num_inference_steps=steps, strength=strength, guidance_scale=scale, model="schnell")
#     surface.show()


    
