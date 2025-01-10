# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object.
base_options = python.BaseOptions(model_asset_path="models\\efficientnet_lite0.tflite")
options = vision.ImageClassifierOptions(base_options=base_options, max_results=3)
classifier = vision.ImageClassifier.create_from_options(options)

from fastapi import FastAPI, UploadFile

app = FastAPI()

import cv2
import numpy as np

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    
    # STEP 3: Load the input image.
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    cv_mat = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)

    # STEP 4: Classify the input image.
    classification_result = classifier.classify(image)
    # print(classification_result)

    # STEP 5: Process the classification result. In this case, visualize it.
    top_category = classification_result.classifications[0].categories[0]
    result = f"{top_category.category_name} ({top_category.score:.2f})"
    return {"result": result}
