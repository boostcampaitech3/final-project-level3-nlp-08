# 한국어 대화 요약 생성을 위한 KoBART model

본 모델은 사전 학습된 [SKT-AI/KoBART](https://github.com/SKT-AI/KoBART#release) 모델을 [AI Hub 한국어 대화 요약 데이터셋](https://aihub.or.kr/aidata/30714)로 fine-tuning한 모델입니다. <br>
Fine-tuning된 모델은 huggingface에 업로드 되어있습니다. [link](https://huggingface.co/chi0/kobart-dial-sum)

## Usage
```python
from transformers import BartForConditionalGeneration
model_name = 'chi0/kobart-dial-sum'
model = BartForConditionalGeneration.from_pretrained(model_name)
```

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

### Dataset Info

- Train dataset: 279,992
- Validation dataset: 35,004

**Topic**
```
['개인및관계', '미용과건강', 상거래(쇼핑)', '시사교육', '식음료', '여가생활', '일과직업', 주거와생활', '행사']
```

<br><br>

```
.
├── SBERT
│   ├── ...
│   ├── README.md
│   ├── sbert.py
│   └── __init__.py
├── data_loader
│   ├── ...
│   ├── __init__.py
│   ├── get_datay.py
│   ├── processing.py
│   └── tokenized_data.py
├── logger
│   ├── ...
│   ├── __init__.py
│   └── logger.py
├── model
│   ├── ...
│   ├── __init__.py
│   └── tokenizer.py
├── README.md
├── __init__.py
├── arguments.py
├── inference.py
├── main.py
└── utils.py
```
