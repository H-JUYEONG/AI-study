from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class KoreanTextSummarizer:
    def __init__(self):
        """KoBART 요약 모델 초기화"""
        print("모델 로드 중...")
        self.model_name = "gogamza/kobart-summarization"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        # GPU 설정
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        print("모델 로드 완료!")

    def preprocess_text(self, text):
        """텍스트 전처리: 문장 분리 및 요약 모델 입력용으로 정리"""
        sentences = text.split("\n")
        processed_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return " ".join(processed_sentences[:5])  # 최대 5문장만 사용

    def summarize(self, text):
        """한국어 텍스트 요약"""
        try:
            # 입력 텍스트 전처리
            text = self.preprocess_text(text)

            # 입력 텍스트 토크나이징
            inputs = self.tokenizer(
                text,
                max_length=1024,
                truncation=True,
                return_tensors="pt",
                padding="max_length",
            ).to(self.device)

            # 요약 생성
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=128,  # 요약 최대 길이
                min_length=30,  # 요약 최소 길이
                length_penalty=1.0,  # 길이 제약 완화
                num_beams=8,  # 탐색 폭 증가
                no_repeat_ngram_size=3,  # 반복 방지
                early_stopping=True,
            )

            # 요약 결과 디코딩
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return self.postprocess_summary(summary.strip())
        except Exception as e:
            return f"요약 중 오류가 발생했습니다: {str(e)}"

    def postprocess_summary(self, summary):
        """요약 결과 후처리"""
        summary = summary.replace("특히 특히", "특히").replace("  ", " ")
        if not summary.endswith("."):
            summary += "."
        return summary


def main():
    # 요약기 초기화
    summarizer = KoreanTextSummarizer()

    # 테스트 텍스트
    korean_text = """
    인공지능 기술은 현대 사회에서 큰 변화를 이끌고 있습니다. 
    특히 자연어 처리 기술은 번역, 요약, 질의응답 시스템 등 다양한 분야에서 실용화되고 있습니다. 
    이러한 발전은 학계와 산업계 모두에서 중요한 연구 주제로 자리 잡고 있습니다.
    """

    print("원본 텍스트:")
    print(korean_text.strip())
    print("\n요약:")
    print(summarizer.summarize(korean_text))


if __name__ == "__main__":
    main()
