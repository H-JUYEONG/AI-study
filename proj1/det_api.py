# STEP 1: Import the necessary modules.
# 필요한 모듈들을 가져옵니다.
import cv2  # OpenCV는 이미지 및 비디오 처리 라이브러리입니다.
import numpy as np  # Numpy는 배열 및 행렬 처리를 위한 라이브러리입니다.
import mediapipe as mp  # Mediapipe는 머신러닝 모델 및 미디어 처리 라이브러리입니다.
from mediapipe.tasks import (
    python,
)  # Mediapipe의 머신러닝 작업에 필요한 Python API를 가져옵니다.
from mediapipe.tasks.python import (
    vision,
)  # Mediapipe의 시각 처리 관련 기능을 가져옵니다.

# STEP 2: Create an ObjectDetector object.
# 객체 감지를 위한 ObjectDetector 객체를 생성합니다.
base_options = python.BaseOptions(
    model_asset_path="models\\efficientdet_lite0.tflite"
)  # EfficientDet Lite 모델의 경로를 설정합니다.
options = vision.ObjectDetectorOptions(
    base_options=base_options, score_threshold=0.5
)  # 객체 감지의 기본 옵션을 설정하고, 점수 임계값을 0.5로 설정합니다.
detector = vision.ObjectDetector.create_from_options(
    options
)  # 위의 옵션을 기반으로 ObjectDetector 객체를 생성합니다.

# STEP 3: Import FastAPI and define the API app.
from fastapi import FastAPI, UploadFile

# FastAPI 앱을 초기화합니다.
app = FastAPI()

# 비동기 API 엔드포인트를 정의합니다.
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    """
    클라이언트가 업로드한 파일에서 객체를 감지하고, 총 객체 수와 '사람(person)' 객체 수를 반환합니다.
    """

    # STEP 3: Load the input image.
    # 업로드된 파일 내용을 읽어서 OpenCV 형식으로 변환합니다.
    contents = await file.read()  # 업로드된 파일의 내용을 비동기로 읽습니다.
    nparr = np.fromstring(contents, np.uint8)  # 파일 내용을 NumPy 배열로 변환합니다.
    cv_mat = cv2.imdecode(
        nparr, cv2.IMREAD_COLOR
    )  # 배열을 OpenCV의 이미지 형식으로 디코딩합니다.
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=cv_mat
    )  # Mediapipe의 이미지 형식으로 변환합니다.

    # STEP 4: Detect objects in the input image.
    # 이미지에서 객체를 감지합니다.
    detection_result = detector.detect(image)  # ObjectDetector를 사용해 객체 감지 수행

    # STEP 5: Process the detection result. In this case, visualize it.
    # 객체 감지 결과를 처리합니다.
    total_count = len(
        detection_result.detections
    )  # 감지된 모든 객체의 수를 계산합니다.

    # 특정 카테고리(여기서는 'person')의 객체 수를 계산합니다.
    person_count = 0
    for detection in detection_result.detections:  # 모든 감지된 객체를 순회합니다.
        if (
            detection.categories[0].category_name == "person"
        ):  # 첫 번째 카테고리가 'person'인 경우를 체크합니다.
            person_count += 1  # 'person' 객체 수를 증가시킵니다.

    # 감지된 객체 총 수와 'person' 객체 수를 결과로 반환합니다.
    result = {"total_count": total_count, "person_count": person_count}

    # 결과를 JSON 형식으로 반환합니다.
    return {"result": result}
