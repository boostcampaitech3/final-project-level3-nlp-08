# 한국어 대화 요약 생성을 위한 (Pretrained) KoBART model

>본 모델은 사전 학습된 [SKT-AI/KoBART](https://github.com/SKT-AI/KoBART#release) 모델을 [AI Hub 한국어 대화 요약 데이터셋](https://aihub.or.kr/aidata/30714)로 fine-tuning한 모델입니다. <br><br>
Fine-tuning된 모델은 huggingface에 업로드 되어있습니다.<br>[Dialogue_Summarization Model](https://huggingface.co/chi0/kobart-dial-sum)


## Usage
```python
from transformers import BartForConditionalGeneration
model_name = 'chi0/kobart-dial-sum'
model = BartForConditionalGeneration.from_pretrained(model_name)
```

***

# CALL-E Project - Dialogue Summarization
## Requirements
***
* PyTorch >= 1.7.1(1.7.1 recommended)
* Python >= 3.8.5(3.8.5 recommended)
* sentence-transformers==2.2.0(Optional for `SBERT`)
* wandb == 0.12.17
* tqdm >= 4.51.0

<br>

## Features
* HyperParameter can be modified via `.yaml` file
* `train.py` Use Huggingface's Trainer Class to Train
* Initially, no files exist in cache_data, so SBERT learning and importing data takes additional time
* `yaml` has the best HyperParameter value set to Base when our team was experimenting

<br>

## Folder Structure
***
```
.
├── SBERT - Training Models to Use as Metrics(Cosine Similarity)
│   ├── ...
│   ├── README.md
│   ├── sbert.py
│   └── __init__.py
│
├── data_loader - anything about data loading goes here
│   ├── ...
│   ├── README.md
│   ├── __init__.py
│   ├── get_data.py
│   ├── processing.py
│   └── tokenized_data.py
│
├── logger - module for logging
│   ├── ...
│   ├── __init__.py
│   └── logger.py
│
├── tokenizer - Tuned Tokenizer for error handling
│   ├── ...
│   ├── README.md
│   ├── __init__.py
│   └── tokenizer.py
│
├── data - default directory for storing input data
│   ├── ...
│   └── README.md
│
├── cache_data - default directory for Trained models & Cached data
│   └── ...
│
├── __init__.py
│
├── README.md
│
├── requirements.txt - List of Required Libraries
│
├── arguments.py - module for switching yaml to HyperParameter
│
├── inference.py - evaluation of trained model
│
├── train.py - main script to start training
│
├── utils.py - small utility functions
│
├── configs.yaml - holds configuration for training
│
└── inference_configs.yaml - holds configuration for evaluation

```

<br>

## Usage
***
### Installing required libraries
* `pip install -r requirements.txt`

<br>

### Train & Evaluation Execution Code
* Train : `python train.py`

* Evaluation : `python inference.py`

<br>

### Config File Format
* Config For Train

```yaml
TrainingArguments:
    output_dir: './results'

    # Setting
    do_train: True
    do_eval: True
    do_predict: False
    predict_with_generate: True
    save_total_limit: 3
    fp16: True
    fp16_opt_level: 'O1'
    seed: 42
    save_steps: 1000
    load_best_model_at_end: False
    report_to: 'wandb'
    metric_for_best_model: 'STS'
    greater_is_better: True

    # HyperParameter
    num_train_epochs: 5
    learning_rate: 5.0e-05
    warmup_steps: 500
    weight_decay: 0.1
    label_smoothing_factor: 0
    lr_scheduler_type: 'linear'

    # Batch Size
    per_device_train_batch_size: 4
    gradient_accumulation_steps: 16
    per_device_eval_batch_size: 16

    # Logging
    logging_dir: "./logs"
    logging_steps: 1000

    # Evaluation
    evaluation_strategy: 'steps' # epoch, no, steps
    eval_steps: 1000

ModelArguments:
    model_name_or_path: 'gogamza/kobart-base-v2'
    cache_dir: './cache_data/model.pt'
    use_fast_tokenizer: True
    use_auth_token: False
    resize_position_embeddings: False
    use_checkpoint: False

EvalModelArguments:
    eval_model_path: './cache_data/RDASS.pt'
    eval_pretrained: 'klue/roberta-base'

    eval_train_path: './data/KorSTS/sts-train.tsv'
    eval_test_path: './data/KorSTS/sts-dev.tsv'

    batch_size: 32
    epoch: 5


DataTrainingArguments:
    train_file: "./data/Training"      # train data path
    validation_file: './data/aihub_valid.json'  # vaild data path
    # test_file: './data/aihub_valid.json'        # test data path

    overwrite_cache: False
    saved_data_path: "./cache_data/raw_data.pickle"
    preprocessing_num_workers: null

    max_source_length: 512
    max_target_length: 64

    val_max_target_length: 64
    pad_to_max_length: True

    max_train_samples: null    
    max_eval_samples: null      
    max_predict_samples: null   

    num_beams: 5               
    ignore_pad_token_for_loss: True
    forced_bos_token: null    

WandbArguments:
    project: "Final Project"
    entity: "miml"
    name: 'Last Model'
```

<br>

* Config for Evaluation

```yaml
ModelArguments:
    model_name_or_path: 'chi0/kobart-dial-sum'
    resize_position_embeddings: False

DataTrainingArguments:
    test_file: './data/aihub_valid.json'        # test data path
    max_source_length: 512

GenerateArguments:
    num_beams: 5
    max_length: 64
    top_k: 50
    top_p: 0.95
    no_repeat_ngram_size: 3
    temperature: 0.7
```