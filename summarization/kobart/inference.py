from logger.logger import *
from datasets import load_dataset
from transformers import (
    set_seed,
)

from arguments import *
from utils import return_model_and_tokenizer
from data_loader.processing import *

def generate_summary(test_samples, model, tokenizer, data_args:DataTrainingArguments, gen_args:GenerateArguments):
    inputs = tokenizer(
        test_samples["dialogue"],
        padding="max_length",
        truncation=True,
        max_length=data_args.max_source_length,
        return_tensors="pt",
    )

    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    outputs = model.generate(input_ids,
                             num_beams=gen_args.num_beams,
                             max_length=gen_args.max_length,
                             attention_mask=attention_mask,
                             top_k=gen_args.top_k,
                             top_p=gen_args.top_p,
                             )

    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return output_str

def main():
    logger = get_logger('Prediction')

    model_args, data_args, gen_args = return_inference_config()

    set_seed(42)
    
    # Return Tokenizer and Model
    tokenizer, model = return_model_and_tokenizer(logger, model_args=model_args, data_args=data_args)

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


    predictions = generate_summary(predict_dataset, model, tokenizer, data_args, gen_args)

    predictions = postprocess_text_first_sent(predictions)

    print(predictions)

if __name__ == "__main__":
    main()