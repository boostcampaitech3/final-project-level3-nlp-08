# Fine-Tuning Data
# KoBART Fine-Tuning Data
## Dataset Info
- Train dataset: 279,992

- Validation dataset: 35,004

- Topic
```
['개인및관계', '미용과건강', 상거래(쇼핑)', '시사교육', '식음료', '여가생활', '일과직업', 주거와생활', '행사']
```

<br>

## Dataset (AI Hub 한국어 대화 요약)
### Dataset 구조
```
.
├── header
│   ├── dialogueInfo
│   │   ├── dialogueID
│   │   ├── numberOfParticipants
│   │   ├── numberOfUtterances
│   │   ├── numberOfTurns
│   │   ├── type
│   │   └── topic
│   └── participantsInfo
│       ├── participantsID
│       ├── gender
│       ├── age
│       └── residentialProvince
└── body
    ├── dialogue
    │   ├── utteranceID
    │   ├── turnID
    │   ├── participantID
    │   ├── date
    │   ├── time
    │   └── utterance
    └── summary
```
***
# SBERT Fine-Tuning Data
## Dataset ([KorSTS](https://github.com/kakaobrain/KorNLUDatasets))

두 문장 사이의 의미적 유사도(Semantic Similarity)를 평가하기 위한 데이터셋<br>
Kakaobrain에서 공개한 KorSTS 데이터셋을 이용하여 모델 학습<br>


|KorSTS |Total  |Train  |Dev. |Test |
|-      |-      |-      |-    |-    |
|Source |	-	    |STS-B  |STS-B|STS-B|
|Translated by|	-	|Machine|	Human|	Human|
|# Examples|	8,628|	5,749|	1,500|	1,379|
|Avg. # words|	7.7|	7.5|	8.7|	7.6|
