from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image

if __name__ == "__main__":
    # 로컬 이미지 파일 직접 열기
    image = Image.open("test.jpg")

    # 이미지 프로세서와 모델 초기화
    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50", revision="no_timm"
    )
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50", revision="no_timm"
    )

    # 이미지 처리
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # 결과 처리
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.9
    )[0]

    # 결과 출력
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
