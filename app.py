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

app = FastAPI()

# URLs for the models on Google Drive
shape_model_url = "https://drive.google.com/uc?id=1JnAi2DVwt0_XbVpRcpVN3pQgR_8xGHuK"
gender_model_url = "https://drive.google.com/uc?id=1MWaJ6hcF9xWfW3zSkM3V9Mgxopq18R3z"

# Local paths to save the models
shape_model_path = "Models/shape.h5"
gender_model_path = "Models/gender.h5"


def download_model_from_google_drive(url, local_path):
    if not os.path.exists(local_path):
        gdown.download(url, local_path, quiet=False)

# Download models if they do not exist locally
download_model_from_google_drive(shape_model_url, shape_model_path)
download_model_from_google_drive(gender_model_url, gender_model_path)

# Load the pre-trained models
shape_model = load_model(shape_model_path)
gender_model = load_model(gender_model_path)

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
        preprocessed_shape_image, preprocessed_gender_image = Functions.preprocess(image)

        # Make predictions using the loaded models
        shape_predictions = Functions.predict_shape(preprocessed_shape_image, shape_model)
        gender_predictions = Functions.predict_gender(preprocessed_gender_image, gender_model)

        # Extract skin tone palette
        skin_tone_palette = Functions.extract_skin_tone(image)

        return JSONResponse(
            content={
                "forma": shape_predictions[0],
                "genero": gender_predictions[0],
                "tono_piel": skin_tone_palette,
            }
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)