import imageio
import numpy as np
import os

# Directory containing the images
image_dir = "output/07"

# Output movie file name
output_file = f"{image_dir}/movie.mp4"

# Get a list of image file names in the directory
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

# Create a writer object to write the movie file
writer = imageio.get_writer(output_file, fps=30)

# Read the first image to get dimensions
first_image = imageio.imread(os.path.join(image_dir, image_files[0]))
#height, width = first_image.shape[:2]
height, width = 512, 800

# Create a frame with the same dimensions as the original images
frame = np.zeros((height, width, 3), dtype=np.uint8)

print(f"Image shape: {first_image.shape}")
print(f"Frame shape: {frame.shape}")

# Iterate over the image files and create scrolling effect
surface = np.copy(frame)

for i in range(0, len(image_files)):
    image_file = image_files[i]

    image_path = os.path.join(image_dir, image_file)
    image = imageio.imread(image_path)
    img_h, img_w, _ = image.shape
    #if image.shape != (512,2048,3):
    if img_w != 1024:
        continue
    
    print(f"Processing file {image_file}")

    # # If it's not the first frame, scroll the previous image
    # if i > 0:
    #     for j in range(width // 2):
    #         # Crop the current image to fit the right half of the frame
    #         cropped_image = image[:, :width - j]
            
    #         # Combine the scrolled previous image with the cropped current image
    #         scrolled_frame = np.roll(frame, -j, axis=1)
    #         scrolled_frame[:, -cropped_image.shape[1]:] = cropped_image


    rate = 128
    for j in range(0, rate):
        rate_dx = img_w//rate
        x = j * rate_dx
        # draw the image to the surface
        #surface = np.copy(frame)
        #surface[:,:] = image[:,x:x+width]
        
        # scroll the surface to the left
        surface = np.roll(surface, -rate_dx, axis=1)
        surface[:,-rate_dx:] = image[:,x:x+rate_dx]
        
        writer.append_data(surface)

# Close the writer to finalize the movie file
writer.close()

print(f"Movie created: {output_file}")