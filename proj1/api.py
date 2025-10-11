from fastapi import FastAPI, UploadFile

app = FastAPI()

# 동기
# @app.post("/files/")
# async def create_file(file: Annotated[bytes, File()]):
#     return {"file_size": len(file)}

# 비동기
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):

    contents = await file.read()
    return {"filename": file.filename,
            "filesize": len(contentss)}
