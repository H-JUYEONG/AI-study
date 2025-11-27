# STEP 1 : import modules
from transformers import pipeline

# STEP 2 : create inference object
classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")

from typing import Annotated
from fastapi import FastAPI, Form

app = FastAPI()

@app.post("/inference/")
async def inference(text: Annotated[str, Form()]):

    # STEP 3 : X

    # STEP 4 : inference
    result = classifier(text)

    # STEP 5 : post processing
    return {"result": result}
