from PIL import Image, ImageFilter
import pytesseract

# Tesseract 경로 설정 (Windows 사용자만 필요)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 이미지 파일 경로
image_path = "1.jpg"

try:
    # 이미지 열기
    image = Image.open(image_path)

    # ---- 전처리 시작 ----
    # 1. 흑백 변환
    image = image.convert("L")  # Grayscale로 변환
    print("이미지 흑백 변환 완료")

    # 2. 이진화 (Threshold)
    threshold = 140  # 임계값
    image = image.point(lambda x: 0 if x < threshold else 255)
    print("이미지 이진화 완료 (임계값: {})".format(threshold))

    # 3. 노이즈 제거 (필터링)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    print("이미지 노이즈 제거 완료")
    # ---- 전처리 끝 ----

    # 변환된 이미지 저장 (디버깅용)
    processed_image_path = "processed_image.jpg"
    image.save(processed_image_path)
    print(f"전처리된 이미지 저장: {processed_image_path}")

    # OCR 실행 (한글 인식)
    custom_config = r"--psm 6 --oem 3"  # 페이지 세그먼트 모드와 OCR 엔진 모드 설정
    text = pytesseract.image_to_string(image, lang="kor", config=custom_config)
    print("OCR 실행 완료")

    # 결과 출력
    if text.strip():
        print("추출된 텍스트:")
        print(text)
    else:
        print("OCR 결과가 비어 있습니다. 이미지 품질이나 설정을 확인하세요.")

except Exception as e:
    print(f"오류 발생: {e}")