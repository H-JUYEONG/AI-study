import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

def split_into_chunks(text, tokenizer, max_length=1024):
    """텍스트를 모델의 최대 길이에 맞게 청크로 나눕니다."""
    sentences = text.split(". ")  # 문장 단위로 분리
    chunks, current_chunk = [], []
    current_length = 0
    for sentence in sentences:
        tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=False)
        if current_length + len(tokenized_sentence) < max_length:
            current_chunk.extend(tokenized_sentence)
            current_length += len(tokenized_sentence)
        else:
            chunks.append([tokenizer.bos_token_id] + current_chunk + [tokenizer.eos_token_id])
            current_chunk = tokenized_sentence
            current_length = len(tokenized_sentence)
    if current_chunk:
        chunks.append([tokenizer.bos_token_id] + current_chunk + [tokenizer.eos_token_id])
    return chunks

def summarize_long_text(
    text, tokenizer, model, chunk_max_length=1024, summary_max_length=512
):
    """긴 텍스트를 청크로 나눠 요약한 뒤 합칩니다."""
    chunks = split_into_chunks(text, tokenizer, chunk_max_length)
    summaries = []
    for chunk in chunks:
        input_tensor = torch.tensor([chunk])
        summary_ids = model.generate(
            input_tensor,
            num_beams=45,
            max_length=summary_max_length,
            eos_token_id=tokenizer.eos_token_id,
        )
        summary = tokenizer.decode(
            summary_ids.squeeze().tolist(), skip_special_tokens=True
        )
        summaries.append(summary)
    return " ".join(summaries)

def recursive_summarize(text, tokenizer, model, chunk_max_length=1024, summary_max_length=512):
    """다단계로 요약하여 결과를 더 자연스럽게 만듭니다."""
    # 1단계: 청크별 요약
    initial_summary = summarize_long_text(text, tokenizer, model, chunk_max_length, summary_max_length)

    # 2단계: 전체 요약을 다시 요약
    tokens = tokenizer.encode(initial_summary, max_length=chunk_max_length, truncation=True)
    input_ids = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]
    summary_ids = model.generate(
        torch.tensor([input_ids]),
        num_beams=4,
        max_length=summary_max_length,
        eos_token_id=tokenizer.eos_token_id,
    )
    final_summary = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
    return final_summary

def main():
    tokenizer = PreTrainedTokenizerFast.from_pretrained("digit82/kobart-summarization")
    model = BartForConditionalGeneration.from_pretrained("digit82/kobart-summarization")

    text = """
    2024년 나는 개발자를 그만 두었다.
회고글에 개발자를 초점으로 두는 것을 고민했지만 내가 사랑했던 직업이기에 주제로 선정되어도 좋을 것 같다.

velog라는 기술 블로그에 개발자를 그만 두었다는 이야기를 쓰는게 재미있지않나?
오랜만에 velog에 로그인 해보니 이제 막 개발을 시작한 사람들의 이야기, 기술 공부에 대한 이야기가 가득하다.
나는 트렌드를 역행하여 개발을 그만두었던 이야기를 쓰려고 한다.
이게 velog에 쓰는 마지막 글이 될 것 같기도 하다.

2024년에 주로 일어난 일은 아래와 같다.
1. 서비스가 망했다.
2. 희망퇴직을 했다.
3. 새로운 직업을 갖다.

본격적으로 내 삶이 바뀐 것은 하반기 부터이다.
상반기에는 내가 회사를 퇴사하고 새로운 직업을 갖을거라 생각하지도 않았다. ;

서비스가 망하다
서비스가 망할거라는것은 입사하자마자 알았다. 그렇다고 바로 퇴사할 수 있나? 나는 그냥 일개 직원일 뿐이다. 내가 할 수 있는 것을 열심히 하고 그저 일을 위한 일을 했다. 내가 다녔던 회사는 맞는 말을 한다고 들어주는 회사는 아니었다.

회사가 다 그런거지 뭐..
이전에는 이사님 방에 찾아가서 의견도 많이 말하고 그랬었다.
망하는 배에 잠깐 탑승하고 다시 좋은 배로 갈아타야지 하는 심정으로 일단 다녔던 것 같다.

그냥 시간이 지나니까 이표정으로 회사 다녔음 ㅇ..
왜 사람들이 회사 오래 다니면 동태눈으로 다니는지 알 것 같다.
결국 나는 대표나 임원들의 자아 실현을 위한 도구일 뿐이고,, 나는 자아를 버리고 그냥 눈에 보이는 결과만 뽑는 개발자 역할을 할 뿐이었다.

어느 순간 회사란 아니 회사를 넘어서 모든 사람들은 그냥 역할놀이를 하는게 아닐까..? 이런 생각이 들었다.
그런말 들어보지 않았나? 도라이 법칙이라고 도라이가 나가면 도라이가 또 들어온다. 근데 도라이가 없으면? 도라이가 나라는 법칙.. 나는 이 말이 우리 세상은 전부 역할놀이 중이라는 것 같았다.
나는 그냥 주어진대로 일하고 정시퇴근 하는 역할~

2024년은 이상하게 다이어리를 쓰지 않아서 뭔생각으로 회사를 다녔는지 기억이 안난다.
매일 작성했던 문서를 다시 확인해보니 당첨자 쿼리를 짜느라 고생했던 것 같다.
상반기 까지는 기능 개발과 유지보수를 많이 했는데 나는 주로 어드민 툴을 맡아서 개발했었다.

뭐 프로젝트 실적이야 좋았던 적이 없었다. 안좋았는데 점점 떨어지기만 했다.
새로운 기능을 한다고 할때마다 음. 이게 되나..? 이랬던 것 같다.

2년동안 이상한 기능개발을 하다보니 프로젝트 인원은 절반이 나갔고 나머지 절반은 전부 계약직분들로 채워졌는데 나는 계약직이나 정직원이나 내 일을 줄일수만 있다면 상관 없다. 문제는 그분들이 계속 나가고 계속 인수인계되고 문서 포맷도 계속 바뀌고 히스토리가 존재하지 않다보니 계속 설명했다.

제일 최악이었던것은 나간분의 기획 문서가 엉망 징창이고 나는 더이상 기획에 대한 반박을 하지 않아서 그냥 기획서대로 개발했더니 다시 무한 재수정..~!!!!!!!!! 결국 몇페이지나 새로 만들었다.

나는 망한 프로젝트, 망한 회사에 정말 많이 있어봤는데 프로젝트 이야기를 좀만 들어봐도 망할지 안망할지 100프로 알 수 있게 되었다.

그리고 이 회사에서 처음으로 프로젝트를 기획단계부터 종료까지 가는 것을 보았는데 어떤 회의나 결정사항에 있어서 '아 저렇게 결정하는건 좋은거구나!'가 아닌 '저런식으로 하지 말아야겠다.' 를 배우는 순간이 더 많았다. 아무튼 많이 배우긴 했다.
프로젝트 굿즈도 받고 유저와의 만남이라는 재미있는 이벤트도 했었는데 유저랑 소통할 수 있었던 점은 참 재미있었다.

그렇게 서비스가 망하고 작년에 이어 올해 또 희망퇴직이 열렸다.
우스갯소리로 이렇게 매년 희망퇴직이 열리면 이건 그냥 이벤트라는 이야기도 했다.
서비스가 망한다는 소문이 점차점차 커졌고 블라인드에를 보면 불안감만 가득했다. 그때쯔음 우리 서비스 UT 분석을 하고, 개선하기 위해 다같이 회의도 했지만 아마 시간끌기 위한 쌩쇼였던 것같다. 아무튼 조직은 망했고 다른 계열사/조직으로 이동신청해서 면접보거나 희망퇴직을 하거나 세가지 선택이 있었다. 나는 희망퇴직 + 타계열사 서류를 넣었지만 시원하게 탈락했고 그냥 희망퇴직을 했다.

작년 희망퇴직 신청을 받았을 때 부터 이직 준비를 해야지 했지만 이전에 이직을 너무 많이 했고 단 1년이라도 그냥 마음 편히 다니고 싶었다. 그래서 이직을 마구마구 미루게 되었다. 몇군데 면접을 보고 했지만 대기업은 서탈이나 면탈이고 스타트업은 연봉이 안맞아서 못갔다.

그쯤 다른 IT 기업에서도 희망퇴직이나 권고사직이 많이 이뤄졌고 나도 갑자기 권고사직이 될 수 있겠다라는 불안감도 들었다. 동시에 개발 일정에 대한 압박감을 많이 느꼈다. 그래서인지 종종 숨을 잘 못쉬고 심장이 조이는 듯한 느낌을 받았다. 개발 일정이 널널해도 그런 느낌을 지속적으로 받았다. 이런 증상이 있는 것을 알았지만 대수롭게 생각하지 않았다. 그냥 내가 기관지가 좀 안좋아서 그런가 보다.. 이러고 말았는데 다시 생각해보니 다 스트레스 때문이었던 것 같다. 이러다가 갑자기 죽을 수도 있겠구나 라는 생각도 들었다.

그리고 예전부터 계속 마음에 걸렸던 생각이 있다. 내가 당장 죽어도 후회는 없을까?
나는 현재 20대 후반이다. 세월호 참사와 이태원 참사를 겪으며 또래 친구들이 많이 죽는 것을 보았다.
이태원 참사 때는 자고 일어났더니 수백통의 전화와 문자들이 와있었다.
그때 가슴이 철렁했다. 나는 원래 이태원에 가려고 했었고 몇 해 전부터 꾸준히 할로윈을 즐겼던 사람이었다.
내 주변 사람들은 그런 나를 잘 알고있었고 걱정이 되어 엄청나게 연락을 했던 것이다.
그날은 몸 상태가 좋지 않아 아쉬운 마음을 가지고 일찍 잠이 들었는데 그런 일이 났던 것이다.

그 뒤로 나는 오늘만 잘 살자. 내일도 말고 일주일도 아니고 오늘을 잘 보내자라는 생각이 조금씩 커졌다.
    """

    summarized_text = recursive_summarize(text, tokenizer, model)
    print(summarized_text)

if __name__ == "__main__":
    main()
