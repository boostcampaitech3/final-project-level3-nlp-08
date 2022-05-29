import torch
from datasets import load_dataset
from transformers import (
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    set_seed,
    BartForConditionalGeneration,
    Seq2SeqTrainingArguments
)
from sentence_transformers import SentenceTransformer, models, util

from logger.logger import *

from arguments import *
from utils import return_model_and_tokenizer
from data_loader.processing import *

import yaml

def main():
    logger = get_logger('Prediction')

    with open('./inference_configs.yaml') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    model_args = configs['ModelArguments']
    data_args = configs['DataTrainingArguments']

    model_args = ModelArguments(**model_args)
    data_args = DataTrainingArguments(**data_args)
    

    set_seed(42)
    
    # Return Tokenizer and Model
    tokenizer, model = return_model_and_tokenizer(logger=logger, model_args=model_args, data_args=data_args)

    # Load Data
    raw_dataset = load_dataset(
        data_args.data_file_type,
        data_files = {'test': data_args.test_file},
        field='data'
    )
    predict_dataset = raw_dataset['test']

    padding = "max_length" if data_args.pad_to_max_length else False
    predict_dataset = predict_dataset.map(
            lambda example: inference_preprocess_function(examples=example,
                                                tokenizer=tokenizer,
                                                max_source_length=data_args.max_source_length,
                                                max_target_length=data_args.max_target_length,
                                                padding=padding),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=predict_dataset.column_names
        )

    # Make predictions
    # predict_dataset 현재 batch 안된 상태
    # Data 입력받는 방식에 따라 입력하는 데이터 형태 변경해야할 예정
    # 현재 데이터는 validation set의 초반 16개만 입력받음
    # 안잘라주면 OOM 문제 발생
    # 입력 방식 확정되면 수정 필요
    output = model.generate(
        inputs = torch.tensor(predict_dataset["input_ids"][:16]),
        attention_mask = torch.tensor(predict_dataset["attention_mask"][:16]),
        top_k=50,
        top_p=0.95,
        max_length = 64
    )
    predictions = tokenizer.batch_decode(
                    output, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
    
    predictions = [pr[:pr.index('.')+1] if '.' in pr else pr for pr in predictions]
    print(predictions)

if __name__ == "__main__":
    main()