# Load and Process Train Dataset

### get_data.py
- json형태의 dataset 불러와서 학습에 필요한 feature만 추출하여 데이터셋 생성
  - `utterance`, `participantID`, `summary`, `dialogueID`만 가져옴
  - 각각의 utterance에 participantID 부착해서 발화자 구분 지음
<br>

### processing.py
- `preprocess`: tokenizer 활용하여 데이터 모델에 맞는 형태로 tokenize
- `postprocess`: 생성된 prediction의 첫 문장만 사용하며 마지막에 `\n` 붙임
<br>

### tokenized_data.py
- tokenizer를 이용하여 sentence encoding
