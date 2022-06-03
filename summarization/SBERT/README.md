# Sentence BERT(SBERT)
## 
KoBART fine-tuning시 성능 평가 지표로 사용하기 위한 모델
  * [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)

<br>

## Issue
* Golden Summary와 생성된 요약문 사이의 겹치는 n-gram만을 고려하는 ROUGE score로는 생성 요약 모델의 성능을 판단할 수 없다고 생각

  * 특히 한국어에 대해서는 ROUGE Score의 Metric으로써의 부정확성이 커진다고 생각이 듬
  
* [Rouge Score의 문제점이 명시된 사이트](https://kakaoenterprise.github.io/deepdive/210729)
> ROGUE의 한계는 한국어 문서 요약 태스크에서 더 도드라져 보입니다. 어근3에 붙은 다양한 접사4가 단어의 역할(의미, 문법적 기능)을 결정하는 언어적 특성을 갖춘 한국어에서는 복잡한 단어 변형이 빈번하게 일어나기 때문이죠. [그림 2]는 드라마 ‘슬기로운 의사생활’ 기사를 요약한 정답 문장과 모델이 생성한 요약 문장을 비교한 예시입니다. ROGUE는 정답 문장과 비교해 철자가 겨우 3개만 다른 오답 문장에 더 높은 점수를 부여합니다.

<br>

## Solution
* Kakao Enterprise AI Research에서 제안한 새로운 지표 [RDASS](https://arxiv.org/abs/2005.03510)에서 아이디어를 가져옴
  * Reference and Document Aware Semantic Evaluation Methods for Korean Language Summarization

* SBERT를 활용한 STS(Semantic Textual Similarity) 문제를 푸는 방식으로 Metric을 지정
  * 사진 출처 : https://wikidocs.net/156176

<div align='center'>

![image](https://user-images.githubusercontent.com/72785706/171684864-2a47a3d8-bf8d-43f4-aba6-0a20abb4d9f7.png)
</div>

* RDASS를 Metric으로써 그대로 활용하지 않은 이유
> 기존의 RDASS는 `prediction`-`document`의 cosine similarity, `prediction`-`golden summary`의 cosine similarity의 평균 값을 활용하여 모델의 성능을 평가<br><br> 여기서 생기는 문제점은 Dialogue Summarization은 애초에 `prediction`-`document`의 Cosine Similarity가 높지 않아 평균 값도 자연스럽게 낮아짐<br>
즉, Dialogue와 요약 문장간의 유사도는 높게 평가되지 않을 것이라 판단.<br><br>
따라서 생성된 요약 문장과 정답 요약 문장 간의 cosine similarity만을 이용해서 점수를 구하고 해당 점수를 모델 성능 평가 지표로 사용하기로 함.

<br>

***

## SBERT Directory의 필요성
SBERT는 STS 데이터셋을 활용하여 Fine-Tuning을 시켜야 실제로 활용할 수 있게 된다.

따라서, Metric으로 활용하기 위해 SBERT를 Fine-Tuning 시켜야 할 필요성이 존재하였고, 이 코드를 해당 디렉터리에 담아 학습을 진행시켰다.