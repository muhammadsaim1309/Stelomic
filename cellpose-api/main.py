from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from segment import run_segmentation_from_bytes

app = FastAPI()

@app.post("/segment/")
async def segment_image_file(file: UploadFile = File(...)):
    image_bytes = await file.read()
    polygons = run_segmentation_from_bytes(image_bytes)
    return JSONResponse({"cell_polygons": polygons})
