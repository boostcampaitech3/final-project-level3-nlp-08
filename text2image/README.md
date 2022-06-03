# Dialogue to Image를 위한 (Pretrained) minDALL-E 모델 

>본 모델은 사전 학습된 [minDALL-E](https://github.com/kakaobrain/minDALL-E) 모델을 Pixabay Custom Dataset으로 Fine-tuning한 모델입니다.


## Usage
```python
from dalle.models import Rep_Dalle
model,_ = Rep_Dalle.from_pretrained(model_path)
```

***

# CALL-E Project - Text to Image
## Requirements
- ```pip install -r requirements.txt```
<br>
## Dataset
### Crawling
```shell
    python crawling/pixabay.py
```

<br>

## Features
* HyperParameter can be modified via `.yaml` file
* `train.py` Fine-tuning Pretrained minDALL-E with Custom Dataset
* `yaml` has the best HyperParameter value set to Base when our team was experimenting

<br>

## Metric
### FID
```shell
python metirc/FID.py --path1 {origin_path} --path2 {target_path}
```
### Clip-Score
ex) model: lafite
```shell
python metirc/clip-score.py --txt {sentence} --num {int number}
```

## Folder Structure
```
.
text2image/
    ├── Readme.md
    ├── crawling/
    │   └── pixabay.py
    ├── metric/
    │   └── FID.py
    └── model/
        ├── README.md
        ├── configs/
        │   └── CALL-E.yaml
        ├── img_data/
        │   ├── illustrations/
        │   ├── scenery/
        │   └── vectors/
        ├── setup.cfg
        ├── test/
        │   ├── figures/
        │   └── sampling_ex.py
        ├── tf_model/
        │   └── model/
        └── dalle/
            ├── LICENSE
            ├── LICENSE.apache-2.0
            ├── LICENSE.cc-by-nc-sa-4.0
            ├── __init__.py
            ├── inference.py
            ├── train.py
            ├── translate.py
            ├── data_loader/
            │   ├── __init__.py
            │   ├── dataloader.py
            │   ├── dataset.py
            │   └── utils.py
            ├── inference.py
            ├── logger/
            │   ├── __init__.py
            │   └── logger.py
            ├── models/
            │   ├── __init__.py
            │   ├── stage1/
            │   ├── stage2/
            │   └── tokenizer.py
            ├── train.py
            ├── translate.py
            └── utils/
                ├── __init__.py
                ├── config.py
                ├── sampling.py
                └── utils.py
        

```

<br>

## Usage
***
### Installing required libraries
* `pip install -r requirements.txt`

<br>

### Train & Evaluation Execution Code
* Train : `python model/dalle/train.py`

* Evaluation : `python model/dalle/inference.py -n {num_candidates} --prompt {sentence}`

<br>

### Config File Format
* Config For Train
```
model/configs/CALL-E.yaml
```

