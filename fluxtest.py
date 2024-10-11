from diffusers import FluxPipeline, FluxInpaintPipeline
import torch
import matplotlib.pyplot as plt

ckpt_id = "black-forest-labs/FLUX.1-schnell"
prompt = "Pixelart 2D side-scroller game image of a house"
height = 400
width = 800

# Initialize the pipeline for denoising
pipe = FluxPipeline.from_pretrained(
    ckpt_id,
    torch_dtype=torch.bfloat16,
)
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
pipe.enable_sequential_cpu_offload()

# Generate the image
image = pipe(
    prompt,
    num_inference_steps=10,
    guidance_scale=0.3,
    height=height,
    width=width,
    max_sequence_length=512
).images[0]

# Print memory usage
print('Max mem allocated (GB) while denoising:', torch.cuda.max_memory_allocated() / (1024 ** 3))

# Display the generated image
plt.imshow(image)
plt.axis('off')  # Hide axis
plt.show()
