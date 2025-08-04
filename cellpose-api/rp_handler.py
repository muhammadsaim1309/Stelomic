import base64
import requests
from segment import run_segmentation
import runpod

def handler(event):
    input_data = event.get("input", {})
    
    base64_image = input_data.get("image_base64")
    image_url = input_data.get("image_url")

    if not base64_image and not image_url:
        return {"error": "Provide either 'image_base64' or 'image_url'."}

    try:
        # Download image if image_url is provided
        if image_url:
            response = requests.get(image_url)
            if response.status_code != 200:
                return {"error": f"Failed to download image. Status code: {response.status_code}"}
            image_bytes = response.content
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Run segmentation on the base64 image
        polygons = run_segmentation(base64_image)
        return {"cell_polygons": polygons}
    except Exception as e:
        return {"error": str(e)}

# Start RunPod handler
runpod.serverless.start({"handler": handler})
