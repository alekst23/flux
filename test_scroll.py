from image_scrolling_inpaint import make_image, scroll_image_left, generate_mask, make_image_inpaint

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
    # Generate the initial image
    initial_image = make_image(prompt, width, height, num_inference_steps, guidance_scale)
    
    # Scroll the image to the left by 50%
    scrolled_image = scroll_image_left(initial_image)
    
    # Generate a mask for the image
    mask = generate_mask(scrolled_image)
    
    # Inpaint the scrolled image using the mask
    inpainted_image = make_image_inpaint(prompt, scrolled_image, mask, width, height, guidance_scale, num_inference_steps)
    
    # Save the final image
    inpainted_image.save(f"output_{prompt.replace(' ', '_')}.png")

