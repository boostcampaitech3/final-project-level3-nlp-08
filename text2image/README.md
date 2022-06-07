
# CALL-E Project - Text to Image

>본 모델은 사전 학습된 [minDALL-E](https://github.com/kakaobrain/minDALL-E) 모델을 Pixabay Custom Dataset으로 Fine-tuning한 모델입니다.

<br>

## Features
* HyperParameter can be modified via `.yaml` file
* `train.py` Fine-tuning Pretrained minDALL-E with Custom Dataset
* `yaml` has the best HyperParameter value set to Base when our team was experimenting

<br>

## Structure

```
.
text2image
    ├── README.md
    ├── crawling
    │   ├── ...
    │   └── pixabay.py
    ├── metric
    │   ├── ...
    │   └── FID.py
    │   └── CLIP.py
    └── model
        ├── ...
        ├── setup.cfg
        ├── configs
        │   ├── ...
        │   └── CALL-E.yaml
        ├── test
        │   ├── ...
        │   └── sampling_ex.py
        └── dalle
            ├── ...
            ├── LICENSE
            ├── LICENSE.apache-2.0
            ├── LICENSE.cc-by-nc-sa-4.0
            ├── __init__.py
            ├── inference.py
            ├── train.py
            ├── translate.py
            ├── data_loader
            │   ├── ...
            │   ├── __init__.py
            │   ├── dataloader.py
            │   ├── dataset.py
            │   └── utils.py
            ├── logger
            │   ├── ...
            │   ├── __init__.py
            │   └── logger.py
            ├── models
            │   ├── ...
            │   ├── stage1
            │   │   ├── ...
            │   │   ├── __init__.py
            │   │   ├── layers.py
            │   │   └── vqgan.py
            │   ├── stage2
            │   │   ├── ...
            │   │   ├── __init__.py
            │   │   ├── layers.py
            │   │   └── transformer.py
            │   ├── __init__.py
            │   └── tokenizer.py
            ├── train.py
            ├── translate.py
            └── utils
                ├── ...
                ├── __init__.py
                ├── config.py
                ├── sampling.py
                └── utils.py
```

<br>

## Usage
```python
from dalle.models import Rep_Dalle
model,_ = Rep_Dalle.from_pretrained(model_path)
```

### Installing required libraries
* `pip install -r requirements.txt`

### Dataset Crawling
```shell
    python crawling/pixabay.py
```
### Metric

#### FID-Score
```shell
python metirc/FID.py --path1 {origin_path} --path2 {target_path}
```

#### CLIP-Score
```shell
python metirc/clip-score.py --txt {sentence} --path {image_path}
```

### Train & Evaluation Execution Code

* Train : `python model/dalle/train.py`
* Evaluation : `python model/dalle/inference.py -n {num_candidates} --prompt {sentence}`

### Configuration File Format

* For Train
```
model/configs/CALL-E.yaml
```

