import logging
import os

from transformers import BartForConditionalGeneration
from transformers.trainer_utils import get_last_checkpoint

from arguments import ModelArguments, DataTrainingArguments
from model.tokenizer import PTTFwithSaveVocab


def return_checkpoint(logger:logging.Logger, training_args):
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    return last_checkpoint

def return_model_and_tokenizer(logger:logging.Logger, model_args:ModelArguments, data_args:DataTrainingArguments):
    tokenizer = PTTFwithSaveVocab.from_pretrained(model_args.model_name_or_path)
    model = BartForConditionalGeneration.from_pretrained(model_args.model_name_or_path)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # TODO : position embedding 크기 resize
    if model_args.resize_position_embeddings:
        model.resize_token_embeddings(len(tokenizer))
        if (
                hasattr(model.config, "max_position_embeddings")
                and model.config.max_position_embeddings < data_args.max_source_length
        ):
            if model_args.resize_position_embeddings is None:
                logger.warning(
                    "Increasing the model's number of position embedding vectors from"
                    f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
                )
                model.resize_position_embeddings(data_args.max_source_length)
            elif model_args.resize_position_embeddings:
                model.resize_position_embeddings(data_args.max_source_length)
            else:
                raise ValueError(
                    f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                    f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                    f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                    " model's position encodings by passing `--resize_position_embeddings`."
                )

    return tokenizer, model

