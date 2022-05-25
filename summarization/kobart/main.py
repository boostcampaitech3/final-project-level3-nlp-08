import nltk  # Here to have a nice missing dependency error message early on
from datasets import load_metric

from transformers import (
    DataCollatorForSeq2Seq,
    set_seed,
    BartForConditionalGeneration
)
from transformers.trainer_utils import get_last_checkpoint

from arguments import *
from data_loader.get_data import get_raw_data, flatten
from model.tokenizer import *
import datasets.arrow_dataset as da

from logger.logger import *

from data_loader.processing import *


def main():
    logger = get_logger('train')
    model_args, data_args, training_args = return_config()

    set_seed(training_args.seed)
    
    # 모듈화
    last_checkpoint = None
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

    datasets = da.Dataset.from_pandas(get_raw_data(logger=logger))

    raw_datasets = datasets.map(flatten, remove_columns=['id'], batched = True)

    tokenizer = PTTFwithSaveVocab.from_pretrained(model_args.model_name_or_path)
    model = BartForConditionalGeneration.from_pretrained(model_args.model_name_or_path)

    # TODO : position embedding 크기 resize
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

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if model_args.use_t5:
        prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
        dataset_columns = ("dialogue", "summary")

    max_source_length = data_args.max_source_length
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    train_data_txt, validation_data_txt = raw_datasets.train_test_split(test_size=0.1).values()

    if training_args.do_train:
        train_dataset = train_data_txt

        train_dataset = train_dataset.map(
            lambda example:preprocess_function(examples=example,
                                               tokenizer=tokenizer,
                                               max_source_length=max_source_length,
                                               max_target_length=max_target_length,
                                               padding=padding,
                                               use_t5=model_args.use_t5,
                                               prefix=prefix if model_args.use_t5 else None),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=raw_datasets.column_names
        )

    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]

        eval_dataset = eval_dataset.map(
            lambda example: preprocess_function(examples=example,
                                                tokenizer=tokenizer,
                                                max_source_length=max_source_length,
                                                max_target_length=max_target_length,
                                                padding=padding,
                                                use_t5=model_args.use_t5,
                                                prefix=prefix if model_args.use_t5 else None),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=raw_datasets.column_names
        )

    # TODO do_predict 바꾸기
    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=raw_datasets.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    metric = load_metric("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        # prediction, labels 각 문장 끝에 줄바꿈 붙이기
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # prediction tokenizer로 decoding해서 문장으로 바꿈
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # labels에서 -100으로 된 padding 제외
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # Evaluation
    max_length = data_args.val_max_target_length
    num_beams = data_args.num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


    if training_args.do_predict:
        logger.info("*** Predict ***")
        # Predict 시작
        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # TODO: Trainer.is_world_process_zero() 뭔지 알아야할 듯
        if trainer.is_world_process_zero():
            # Generative metric 후 결과 문장 생성
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                # output_dir에 generated_predictions.txt로 결과 저장
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

if __name__ == "__main__":
    main()