from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from image_scrolling_inpaint import make_image
from starlette.responses import StreamingResponse
import io

app = FastAPI()

class ImageRequest(BaseModel):
    description: str
    width: int
    height: int

@app.post("/generate_image")
def generate_image(request: ImageRequest):
    try:
        # Generate the flux image based on the description and size
        img = make_image(
            request.description,
            request.width,
            request.height,
            num_inference_steps=5,
            guidance_scale=0.0,
            model="schnell"
        )

        # Convert the image to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Return the image as a StreamingResponse
        return StreamingResponse(img_byte_arr, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
