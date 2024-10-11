from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from image_scrolling_inpaint import make_image
import io
import os
from uuid import uuid4

app = FastAPI()

# Directory to store generated images
image_dir = "static/images"
os.makedirs(image_dir, exist_ok=True)

# Serve static files from the image_dir
app.mount("/static", StaticFiles(directory="static"), name="static")

class ImageRequest(BaseModel):
    description: str
    width: int
    height: int

@app.post("/generate_image")
def generate_image(request: ImageRequest):
    try:
        print(f"Received request: {request}")
        # Generate the flux image based on the description and size
        img = make_image(
            request.description,
            request.width,
            request.height,
            num_inference_steps=5,
            guidance_scale=0.0,
            model="schnell"
        )

        # Generate a unique filename for the image
        image_filename = f"{uuid4()}.png"
        image_path = os.path.join(image_dir, image_filename)

        # Save the image to the filesystem
        img.save(image_path)

        # Return the URL where the image can be accessed
        image_url = f"/static/images/{image_filename}"
        print(f"Generated image: {image_url}")
        return image_url
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
