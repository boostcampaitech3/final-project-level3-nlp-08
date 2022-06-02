# Sentence BERT

KoBART fine-tuning시 성능 평가 지표로 사용하기 위한 모델<br>
Golden Summary와 생성된 요약문 사이의 겹치는 n-gram만을 고려하는 ROUGE score로는 생성 요약 모델의 성능을 판단할 수 없다고 생각<br>
Kakao Enterprise AI Research에서 제안한 새로운 지표 [RDASS](https://kakaoenterprise.github.io/deepdive/210729)에서 아이디어를 가져옴,<br>
<br>
기존의 RDASS는 `prediction`-`document`의 cosine similarity, `prediction`-`golden summary`의 cosine similarity를 이용해서 모델의 성능을 평가<br>
Dialogue summarization의 특성 상 Dialogue와 요약 문장간의 유사도는 높게 평가되지 않을 것이라 판단.<br>
생성된 요약 문장과 정답 요약 문장 간의 cosine similarity만을 이용해서 점수를 구하고 해당 점수를 모델 성능 평가 지표로 사용하기로 함.<br>
<br>

## Dataset ([KorSTS](https://github.com/kakaobrain/KorNLUDatasets))

두 문장 사이의 의미적 유사도(Semantic Similarity)를 평가하기 위한 데이터셋<br>
Kakaobrain에서 공개한 KorSTS 데이터셋을 이용하여 모델 학습<br>


|KorSTS |Total  |Train  |Dev. |Test |
|-      |-      |-      |-    |-    |
|Source |	-	    |STS-B  |STS-B|STS-B|
|Translated by|	-	|Machine|	Human|	Human|
|# Examples|	8,628|	5,749|	1,500|	1,379|
|Avg. # words|	7.7|	7.5|	8.7|	7.6|
