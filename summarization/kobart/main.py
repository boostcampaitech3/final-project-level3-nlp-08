import wandb
from datasets import load_metric

from data_loader.tokenized_data import return_tokenized_sentence
from utils import return_checkpoint, return_model_and_tokenizer
from SBERT.sbert import *

from transformers import (
    DataCollatorForSeq2Seq,
    set_seed
)

from arguments import *
from data_loader.get_data import return_data
from model.tokenizer import *

from logger.logger import *

from data_loader.processing import *

def main():
    # Setting Log
    logger = get_logger('train')

    # Setting Argument
    model_args, data_args, training_args, wandb_args = return_train_config()

    # Setting Seed
    set_seed(training_args.seed)

    # Bring Checkpoint
    if model_args.use_checkpoint:
        last_checkpoint = return_checkpoint(logger=logger, training_args=training_args)
    else:
        last_checkpoint = None

    # Bring Dataset
    train_data_txt, validation_data_txt = return_data(logger=logger, data_args=data_args)

    # Bring Tokenizer & Model
    tokenizer, model = return_model_and_tokenizer(logger=logger, model_args=model_args, data_args=data_args)

    # Encoding
    if training_args.do_train:
        train_dataset = return_tokenized_sentence(train_data_txt, tokenizer, data_args)
    if training_args.do_eval:
        eval_dataset = return_tokenized_sentence(validation_data_txt, tokenizer, data_args)

    metric = load_metric("rouge")

    eval_model = return_metric_model()

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

        data1 = eval_model.encode(decoded_labels)
        data2 = eval_model.encode(decoded_preds)

        answer_list = []
        for s1, s2 in zip(data1, data2):
            cos_scores = util.pytorch_cos_sim(s1, s2)

            answer_list.append(cos_scores[0])

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        result['STS'] = sum(answer_list)/len(answer_list)
        return result

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    wandb.init(project=wandb_args.project, entity=wandb_args.entity, name= wandb_args.name)

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
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        torch.save(model.state_dict(), model_args.cache_dir)

        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=data_args.val_max_target_length,
                                   num_beams=data_args.num_beams,
                                   metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()