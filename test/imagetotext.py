from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# 로컬 파일 경로를 직접 지정
image_path = "1.jpg"  # 이미지 파일이 있는 정확한 경로를 지정하세요
image = Image.open(image_path).convert("RGB")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]