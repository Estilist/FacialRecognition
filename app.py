from fastapi import FastAPI, File, UploadFile, HTTPException, Form # fastapi, uvicorn, python-multipart
from fastapi.responses import JSONResponse
import cv2 # opencv-python
import numpy as np # numpy
from keras.models import load_model # keras
from Backend.functions import Functions
import requests
from io import BytesIO
import os
import gdown
from azure.storage.blob import BlobServiceClient
import os


app = FastAPI()

BLOB_CONNECTION_STRING = 'DefaultEndpointsProtocol=https;AccountName=estilistimages;AccountKey=F3fyNe9iTvlj+ljIoCrWSJI7YzAU470cu5dnc4wcv5kAqjirvkSVeQBXe4IDH/NmucDG1D5e0rsR+AStIQ7u5A==;EndpointSuffix=core.windows.net'
CONTAINER_NAME = 'models'
BLOB_NAME = 'shape.h5'
LOCAL_MODEL_PATH = 'estilist_backend/Models/shape.h5'

def download_model_from_blob():
    if not os.path.exists(LOCAL_MODEL_PATH):
        blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=BLOB_NAME)

        with open(LOCAL_MODEL_PATH, "wb") as model_file:
            model_file.write(blob_client.download_blob().readall())
        
download_model_from_blob()
shape_model = load_model(LOCAL_MODEL_PATH)

"Accepts an image file or URL and returns the predicted shape, gender, and skin tone palette"
@app.post("/predict/")
async def predict(file: UploadFile = File(None), url: str = Form(None)):
    try:
        if file:
            # Read the uploaded image file
            image_data = await file.read()
        elif url:
            # Decode the URL
            url = url.replace("\\", "")
            
            # Download the image from the URL
            response = requests.get(url)
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to download the image")
            image_data = BytesIO(response.content).read()
        else:
            raise HTTPException(status_code=400, detail="No file or URL provided")

        image_array = np.fromstring(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Preprocess the image
        preprocessed_shape_image = Functions.preprocess(image)

        # Make predictions using the loaded models
        shape_predictions = Functions.predict_shape(preprocessed_shape_image, shape_model)

        # Extract skin tone palette
        skin_tone_palette = Functions.extract_skin_tone(image)

        return JSONResponse(
            content={
                "forma": shape_predictions[0],
                "tono_piel": skin_tone_palette,
            }
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)