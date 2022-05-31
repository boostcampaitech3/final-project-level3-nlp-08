from logger.logger import *
from datasets import load_dataset
from transformers import set_seed
import torch

from arguments import *
from utils import return_model_and_tokenizer
from data_loader.processing import *

def generate_summary(inputs, model, tokenizer, data_args:DataTrainingArguments, gen_args:GenerateArguments):
    inputs = tokenizer(
        inputs["dialogue"],
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = get_logger('Prediction')

    model_args, data_args, gen_args = return_inference_config()

    set_seed(42)
    
    # Return Tokenizer and Model
    tokenizer, model = return_model_and_tokenizer(logger, model_args=model_args, data_args=data_args)
    model = model.to(device)

    # Load Data in json
    raw_dataset = load_dataset(
        'json',
        data_files = {'test': data_args.test_file},
        field='data'
    )
    predict_dataset = raw_dataset['test']

    predictions = generate_summary(predict_dataset, model, tokenizer, data_args, gen_args)
    predictions = postprocess_text_first_sent(predictions)

    # 해당 부분 아래는 Text2Image 코드 실행되어야함
    print(predictions)

if __name__ == "__main__":
    main()