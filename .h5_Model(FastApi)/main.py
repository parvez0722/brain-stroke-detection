from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

origins = [
    '*'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Load the trained model
model = load_model("C:/Brain_stroke/brain_Stroke_modeel.h5")

# Define constants
img_height = 256
img_width = 256

from tensorflow.keras.preprocessing import image

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values to between 0 and 1
    return img_array


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
        # Save the uploaded file
        with open(file.filename, "wb") as image_file:
            image_file.write(file.file.read())

        # Preprocess the image
        img_array = preprocess_image(file.filename)

        # Make prediction
        prediction = model.predict(img_array)

        # Return prediction resul
        result = {
            "res" : prediction[0][0].item()
        }
        return JSONResponse(content=result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
