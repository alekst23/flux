import logging
from PIL import Image
from image_scrolling_inpaint import make_image, make_image_inpaint, new_img_white, new_img_noise
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

width, height = 1024, 1024

#seed = template_noisy_horizon(width, height)
#seed.show()

mask = new_img_white(width, height)

prompt = "Sprite for a 2D platformer, pixelart on white background, a gremlin running with a pouch."
scale = 0.0
steps = 5

#surface = make_image(prompt, width, height, num_inference_steps=steps, guidance_scale=scale, model="schnell")
img = load_image("output/00C.PNG")
img.show()

prompt = "The Next frame of the animation of a gremlin running with a pouch, pixelart on white background."
# surface2 = make_image_inpaint(prompt, surface, mask, width, height, num_inference_steps=12, strength=0.8, model="dev")
# surface2.show()

# surface2 = make_image_inpaint(prompt, surface, mask, width, height, num_inference_steps=8, strength=0.999, model="schnell")
# surface2.show()

for i in range(0, 5):
    scale = 0.0 + 0.2*i
    steps = 12
    strength = 0.5 + 0.1*i
    logging.info(f"GENERATING, scale={scale}, steps={steps}, strength={strength}")

    logging.info("-dev")
    #surface = make_image(prompt, width, height, num_inference_steps=steps, guidance_scale=scale, model="dev")
    surface = make_image_inpaint(prompt, img, mask, width, height, guidance_scale=scale, num_inference_steps=12, strength=strength, model="dev")
    surface.show()

    logging.info("-schnell")
    #surface = make_image(prompt, width, height, num_inference_steps=steps//2, guidance_scale=scale, model="schnell")
    surface = make_image_inpaint(prompt, img, mask, width, height, guidance_scale=scale, num_inference_steps=4, strength=strength, model="schnell")
    surface.show()

# for i in range(0,5):
#     steps = 5+i*2
#     strength = 0.95 + 0.01*i
#     logging.info(f"GENERATING, steps={steps}, strength={strength}")
#     surface = make_image_inpaint(prompt, seed, mask, width, height, num_inference_steps=steps, strength=0.8, model="dev")
#     surface.show()

    
