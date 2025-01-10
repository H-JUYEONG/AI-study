from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch

if __name__ == "__main__":
    # 로컬 이미지 파일 직접 열기
    image = Image.open("test.jpg")

    # 모델과 이미지 프로세서 초기화
    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")
    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

    # 이미지 처리
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # 결과 처리
    logits = outputs.logits
    bboxes = outputs.pred_boxes

    # 결과 출력
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(
        outputs, threshold=0.9, target_sizes=target_sizes
    )[0]

    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
