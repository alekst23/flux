import logging
from PIL import Image
from image_utils import make_image_inpaint, new_img_white, new_img_noise
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def template_horizon(width, height, pos: float=0.6)->Image.Image:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[int(height*pos):, :] = 50
    canvas[:int(height*pos), :] = 255
    return Image.fromarray(canvas, 'RGB')

width, height = 1024, 512

seed = template_horizon(width, height)
i2 = new_img_noise(width, height)
seed.show()

mask = new_img_white(width, height)

prompt = "Pixelart background for a 2D videogame, an alien landscape with futuristic buildings."

surface = make_image_inpaint(prompt, i2, mask, width, height, num_inference_steps=20, strength=0.99999, model="dev")
surface.show()

for i in range(0,5):
    steps = 5+i*2
    strength = 0.95 + 0.01*i
    logging.info(f"GENERATING, steps={steps}, strength={strength}")
    surface = make_image_inpaint(prompt, seed, mask, width, height, num_inference_steps=steps, strength=0.8, model="dev")
    surface.show()

    
