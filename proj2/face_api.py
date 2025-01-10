# STEP 1 : import modules
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image  # 데이터 가져오는 함수

# STEP 2 : create inference object(instance)
face = FaceAnalysis()
face.prepare(ctx_id=0, det_size=(640, 640))

from fastapi import FastAPI, UploadFile
import cv2
import numpy as np

app = FastAPI()

# 비동기
@app.post("/uploadfile/")
async def create_upload_file(file1: UploadFile, file2: UploadFile):

    # STEP 3: Load the input image.
    contents1 = await file1.read()
    contents2 = await file2.read()

    nparr1 = np.fromstring(contents1, np.uint8)
    nparr2 = np.fromstring(contents2, np.uint8)

    img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

    # STEP 4 : inference
    faces1 = face.get(img1)
    faces2 = face.get(img2)
    assert len(faces1)==1
    assert len(faces2)==1

    # STEP 5 : post processing
    # 유사도 결과값이 0.4 이상이면 동일 인물
    face_feat1 = faces1[0].normed_embedding
    face_feat2 = faces2[0].normed_embedding
    face_feat1 = np.array(face_feat1, dtype=np.float32)
    face_feat2 = np.array(face_feat2, dtype=np.float32)
    sims = np.dot(face_feat1, face_feat2.T)
    print(sims)

    if sims > 0.4:
        return {"result" : "동일 인물입니다."}
    else:
        return {"result" : "동일 인물이 아닙니다."}