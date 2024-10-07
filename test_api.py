import requests

# Define the API endpoint URL
api_url = "http://localhost:8000/generate_image"

# Create a JSON payload with the description, width, and height
payload = {
    "description": "A scenic landscape with mountains and a clear sky",
    "width": 512,
    "height": 512
}

# Send a POST request to the endpoint with the payload
response = requests.post(api_url, json=payload)

# Check the response status code and print the result
if response.status_code == 200:
    print("Request successful. Saving the image...")
    # Assuming the response contains the image in binary format
    with open("generated_image.png", "wb") as f:
        f.write(response.content)
    print("Image saved as 'generated_image.png'.")
else:
    print(f"Request failed with status code: {response.status_code}")
    print("Response content:", response.text)
