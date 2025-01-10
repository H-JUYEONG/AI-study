from typing import Annotated
from fastapi import FastAPI, Form

app = FastAPI()

@app.post("/inference/")
async def inference(text: Annotated[str, Form()]):
    
    return {"result": text}
