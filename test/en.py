from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


class EnglishToKoreanSummarizer:
    def __init__(self):
        """영어 요약 및 한국어 번역 모델 초기화"""
        print("모델 로드 중...")
        # 영어 요약 모델
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        # 다국어 번역 모델 (mT5)
        self.mt5_name = (
            "google/mt5-small"  # mT5 모델 이름 (small, base, large 선택 가능)
        )
        self.mt5_tokenizer = AutoTokenizer.from_pretrained(self.mt5_name)
        self.mt5_model = AutoModelForSeq2SeqLM.from_pretrained(self.mt5_name)
        print("모델 로드 완료!")

    def summarize_english(self, text):
        """영어 텍스트 요약"""
        try:
            input_length = len(self.summarizer.tokenizer(text)["input_ids"])
            max_length = max(
                30, int(input_length * 0.6)
            )  # max_length를 입력 길이에 따라 동적으로 조정
            min_length = max(10, int(input_length * 0.3))  # min_length 설정
            summary = self.summarizer(
                text, max_length=max_length, min_length=min_length, do_sample=False
            )
            return summary[0]["summary_text"]
        except Exception as e:
            return f"영어 요약 중 오류: {str(e)}"

    def translate_to_korean(self, text):
        """영어 요약을 한국어로 번역 (mT5 사용)"""
        try:
            # mT5 작업 지시어 추가
            task_text = f"translate English to Korean: {text}"
            inputs = self.mt5_tokenizer(
                task_text, max_length=512, truncation=True, return_tensors="pt"
            )

            # 번역 생성
            outputs = self.mt5_model.generate(
                inputs.input_ids,
                max_length=150,
                min_length=30,
                length_penalty=1.0,
                num_beams=6,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

            # 결과 디코딩
            translation = self.mt5_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            return translation.strip()
        except Exception as e:
            return f"번역 중 오류: {str(e)}"

    def summarize(self, text):
        """영어 텍스트 요약 및 한국어 번역"""
        try:
            # 1단계: 영어 텍스트 요약
            english_summary = self.summarize_english(text)
            if "오류" in english_summary:
                return english_summary
            print(f"영어 요약 결과: {english_summary}")

            # 2단계: 한국어로 번역
            korean_summary = self.translate_to_korean(english_summary)
            return korean_summary
        except Exception as e:
            return f"요약 및 번역 중 오류가 발생했습니다: {str(e)}"


def main():
    # 요약기 초기화
    summarizer = EnglishToKoreanSummarizer()

    # 테스트 텍스트
    english_text = """
    Dogs are loyal and friendly animals that have been living with humans for thousands of years.
    They are often called "man's best friend" because of their companionship and protection.
    Different breeds have different characteristics, making them suitable for various tasks such as hunting, herding, or simply being a loving pet.
    """

    print("원본 텍스트:")
    print(english_text.strip())
    print("\n한국어 요약:")
    print(summarizer.summarize(english_text))


if __name__ == "__main__":
    main()
