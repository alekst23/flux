import logging
from PIL import Image
from image_utils import make_image
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


width, height = 512, 512

prompt = "front view of a residential building in 1905 France, pixel art graphic 2D roguelike orthographic projection."

config_dev = {
    "num_inference_steps": 20,
    "guidance_scale": 0.0,
    "model": "dev",
}
config_schnell = {
    "num_inference_steps": 4,
    "guidance_scale": 0.0,
    "model": "schnell",
}

surface = make_image(prompt, width, height, **config_dev)

surface.show()

# for i in range(0,5):
#     steps = 5+i*2
#     strength = 0.95 + 0.01*i
#     logging.info(f"GENERATING, steps={steps}, strength={strength}")
#     surface = make_image_inpaint(prompt, seed, mask, width, height, num_inference_steps=steps, strength=0.8, model="dev")
#     surface.show()

    
