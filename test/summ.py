from transformers import MBartForConditionalGeneration, MBartTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re


class MultilingualSummarizer:
    def __init__(self):
        print("모델 초기화 중...")

        # mBART - 혼합 텍스트 처리 및 번역용
        self.mbart_name = "facebook/mbart-large-cc25"
        self.mbart_tokenizer = MBartTokenizer.from_pretrained(self.mbart_name)
        self.mbart_model = MBartForConditionalGeneration.from_pretrained(
            self.mbart_name
        )

        # KoBART - 한국어 요약
        self.ko_name = "gogamza/kobart-summarization"
        self.ko_tokenizer = AutoTokenizer.from_pretrained(self.ko_name)
        self.ko_model = AutoModelForSeq2SeqLM.from_pretrained(self.ko_name)

        # BART - 영어 요약
        self.en_name = "facebook/bart-large-cnn"
        self.en_tokenizer = AutoTokenizer.from_pretrained(self.en_name)
        self.en_model = AutoModelForSeq2SeqLM.from_pretrained(self.en_name)

        # GPU 설정
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mbart_model.to(self.device)
        self.ko_model.to(self.device)
        self.en_model.to(self.device)

        print("모델 초기화 완료!")

    def preprocess_text(self, text):
        text = " ".join(text.split())
        return text[:1024]

    def detect_language(self, text):
        korean = len(re.findall("[가-힣]", text))
        english = len(re.findall("[a-zA-Z]", text))
        total = korean + english
        if total == 0:
            return "en"
        ko_ratio = korean / total
        en_ratio = english / total
        if ko_ratio >= 0.3 and en_ratio >= 0.3:
            return "mixed"
        elif ko_ratio > en_ratio:
            return "ko"
        else:
            return "en"

    def summarize_mixed(self, text):
        self.mbart_tokenizer.src_lang = "ko_KR"
        inputs = self.mbart_tokenizer(
            text,
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        forced_bos_token_id = self.mbart_tokenizer.lang_code_to_id["ko_KR"]
        summary_ids = self.mbart_model.generate(
            inputs["input_ids"],
            forced_bos_token_id=forced_bos_token_id,
            max_length=128,
            min_length=30,
            length_penalty=1.5,
            num_beams=5,
            no_repeat_ngram_size=4,
        )
        return self.mbart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def summarize_english(self, text):
        inputs = self.en_tokenizer(
            text,
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        summary_ids = self.en_model.generate(
            inputs["input_ids"],
            max_length=128,
            min_length=30,
            length_penalty=2.0,
            num_beams=5,
            no_repeat_ngram_size=3,
        )
        english_summary = self.en_tokenizer.decode(
            summary_ids[0], skip_special_tokens=True
        )

        # 영어 -> 한국어 번역
        self.mbart_tokenizer.src_lang = "en_XX"
        inputs = self.mbart_tokenizer(
            english_summary,
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        forced_bos_token_id = self.mbart_tokenizer.lang_code_to_id["ko_KR"]
        translation_ids = self.mbart_model.generate(
            inputs["input_ids"],
            forced_bos_token_id=forced_bos_token_id,
            max_length=128,
            min_length=30,
            length_penalty=1.5,
            num_beams=5,
        )
        return self.mbart_tokenizer.decode(translation_ids[0], skip_special_tokens=True)

    def summarize_korean(self, text):
        inputs = self.ko_tokenizer(
            text,
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        summary_ids = self.ko_model.generate(
            inputs["input_ids"],
            max_length=128,
            min_length=30,
            length_penalty=1.5,
            num_beams=8,
            no_repeat_ngram_size=4,
        )
        return self.ko_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def summarize(self, text):
        text = self.preprocess_text(text)
        lang_type = self.detect_language(text)
        if lang_type == "mixed":
            summary = self.summarize_mixed(text)
        elif lang_type == "en":
            summary = self.summarize_english(text)
        else:
            summary = self.summarize_korean(text)
        return summary


def main():
    summarizer = MultilingualSummarizer()
    test_cases = [
        "This project is very exciting! 이 프로젝트는 정말 흥미로워요. AI 기술은 매우 강력합니다.",
        "Artificial Intelligence is revolutionizing our daily lives. It's practical and powerful.",
        "인공지능 기술이 발전하면서 우리의 일상생활이 크게 변화하고 있습니다. 자연어 처리 분야는 실용화되고 있습니다.",
    ]
    for i, text in enumerate(test_cases, 1):
        print(f"\n테스트 케이스 {i}")
        print("원본 텍스트:")
        print(text.strip())
        print("\n요약:")
        print(summarizer.summarize(text))
        print("-" * 50)


if __name__ == "__main__":
    main()
