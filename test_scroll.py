import logging
from PIL import Image
from image_scrolling_inpaint import make_image, scroll_image_left, generate_mask, make_image_inpaint

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define a list of prompts for image generation
prompts = [
    "A scenic landscape with mountains and a clear sky",
    "A futuristic cityscape at night with neon lights",
    "A serene beach with palm trees and a sunset"
]

# Parameters for image generation
width, height = 512, 512
num_inference_steps = 10
guidance_scale = 0.3

# Generate and process images based on prompts
for prompt in prompts:
    logging.info(f"Generating initial image for prompt: '{prompt}'")
    initial_image = make_image(prompt, width, height, num_inference_steps, guidance_scale)
    initial_image.save("output/01.png")
    
    logging.info("Scrolling the image to the left by 50%")
    scrolled_image = scroll_image_left(initial_image)
    # Convert the scrolled image from RGBA to RGB
    scrolled_image = scrolled_image.convert("RGB")
    scrolled_image.save("output/02.png")

    logging.info("Generating mask for the image")
    mask = generate_mask(scrolled_image)
    mask.save("output/03.png")
    
    logging.info("Inpainting the scrolled image using the mask")
    inpainted_image = make_image_inpaint(prompt, scrolled_image, mask, width, height, guidance_scale, num_inference_steps)
    
    # Save or display the final image
    output_filename = f"output/{prompt.replace(' ', '_')}.png"
    inpainted_image.save(output_filename)
    logging.info(f"Saved the final image as '{output_filename}'")
    
    # To display:
    # inpainted_image.show()
