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

width, height = 1024, 512

#seed = template_noisy_horizon(width, height)
#seed.show()

mask = new_img_white(width, height)

prompt = "You are standing before a rock face and see a closed door built into the mountain."
surface = make_image(prompt, width, height, num_inference_steps=5, guidance_scale=0.0, model="schnell")
surface.show()


for i in range(0, 5):
    scale = 0.0 + 0.2*i
    steps = 12
    logging.info(f"GENERATING, scale={scale}, steps={steps}")
    logging.info("-dev")
    surface = make_image(prompt, width, height, num_inference_steps=steps, guidance_scale=scale, model="dev")
    #surface = make_image_inpaint(prompt, seed, mask, width, height, num_inference_steps=12, strength=0.999, model="dev")
    surface.show()

    logging.info("-schnell")
    surface = make_image(prompt, width, height, num_inference_steps=steps//2, guidance_scale=scale, model="schnell")
    surface.show()

# for i in range(0,5):
#     steps = 5+i*2
#     strength = 0.95 + 0.01*i
#     logging.info(f"GENERATING, steps={steps}, strength={strength}")
#     surface = make_image_inpaint(prompt, seed, mask, width, height, num_inference_steps=steps, strength=0.8, model="dev")
#     surface.show()

    
