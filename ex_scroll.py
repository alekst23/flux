import logging
from PIL import Image
from image_utils import make_image, scroll_image_left, generate_mask_half, generate_mask_full, make_image_inpaint

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parameters for image generation
width, height = 512, 512
num_inference_steps = 10
guidance_scale = 0.0

def print_image_properties(image, image_name):
    logging.info(f"Properties of {image_name}:")
    logging.info(f"Mode: {image.mode}")
    logging.info(f"Size: {image.size}")
    logging.info(f"Format: {image.format}")
    logging.info(f"Info: {image.info}")
    if hasattr(image, 'palette') and image.palette:
        logging.info(f"Palette: {image.palette.getdata()}")
    if "icc_profile" in image.info:
        logging.info(f"ICC Profile: Present")
    else:
        logging.info(f"ICC Profile: Not present")


# Generate and process images based on prompts
prompt = "Pixelart background, a scenic landscape with mountains and a clear sky."

logging.info(f"Generating initial image for prompt: '{prompt}'")
initial_image = make_image(prompt, width, height, num_inference_steps, guidance_scale)
initial_image.save("output/01.png")

print_image_properties(initial_image, "Original Image")

logging.info("Scrolling the image to the left by 50%")
scrolled_image = scroll_image_left(initial_image)
scrolled_image = scrolled_image.convert("RGB")
scrolled_image.save("output/02.png")

print_image_properties(scrolled_image, "Scrolled Image")

logging.info("Generating mask for the image")
mask = generate_mask_half(scrolled_image)
mask.save("output/03.png")
mask_full = generate_mask_full(scrolled_image)

logging.info("Inpainting the scrolled image using the mask")
make_image_inpaint(prompt, scrolled_image, mask, width, height, guidance_scale=0.0, num_inference_steps=num_inference_steps, strength=0.880).save("output/04.1.png")
make_image_inpaint(prompt, scrolled_image, mask, width, height, guidance_scale=0.0, num_inference_steps=num_inference_steps, strength=0.890).save("output/04.2.png")
make_image_inpaint(prompt, scrolled_image, mask, width, height, guidance_scale=0.0, num_inference_steps=num_inference_steps, strength=0.899).save("output/04.3.png")
make_image_inpaint(prompt, scrolled_image, mask, width, height, guidance_scale=0.0, num_inference_steps=num_inference_steps, strength=0.900).save("output/04.4.png")
make_image_inpaint(prompt, scrolled_image, mask, width, height, guidance_scale=0.0, num_inference_steps=num_inference_steps, strength=0.910).save("output/04.5.png")

make_image_inpaint(prompt, scrolled_image, mask_full, width, height, guidance_scale=0.0, num_inference_steps=num_inference_steps, strength=0.3).save("output/05.1.png")
make_image_inpaint(prompt, scrolled_image, mask_full, width, height, guidance_scale=0.0, num_inference_steps=num_inference_steps, strength=0.5).save("output/05.2.png")
make_image_inpaint(prompt, scrolled_image, mask_full, width, height, guidance_scale=0.0, num_inference_steps=num_inference_steps, strength=0.6).save("output/05.3.png")
make_image_inpaint(prompt, scrolled_image, mask_full, width, height, guidance_scale=0.0, num_inference_steps=num_inference_steps, strength=0.7).save("output/05.4.png")
make_image_inpaint(prompt, scrolled_image, mask_full, width, height, guidance_scale=0.0, num_inference_steps=num_inference_steps, strength=0.8).save("output/05.5.png")

inpainted_image = make_image_inpaint(prompt, scrolled_image, mask, width, height, guidance_scale=0.1, num_inference_steps=num_inference_steps, strength=0.95)
inpainted_image.save("output/04.png")


logging.info(f"Saved the final image as 'output/inpainted_image.png'")

# To display:
# inpainted_image.show()
