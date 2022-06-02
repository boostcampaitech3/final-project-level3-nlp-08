import pandas as pd
import datasets.arrow_dataset as da
import math
import logging
import os

import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample

from arguments import EvalModelArguments, return_eval_model_config


def return_metric_model():
    model_args = return_eval_model_config()

    if os.path.exists(model_args.eval_model_path):
        pretrained_model_name = model_args.eval_pretrained

        # Load Embedding Model
        embedding_model = models.Transformer(
            model_name_or_path=pretrained_model_name,
            max_seq_length=256,
            do_lower_case=True
        )

        # Only use Mean Pooling -> Pooling all token embedding vectors of sentence.
        pooling_model = models.Pooling(
            embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )

        eval_model = SentenceTransformer(modules=[embedding_model, pooling_model])

        eval_model.load_state_dict(torch.load(model_args.eval_model_path))
    else:
        eval_model = train_model(model_args)

    return eval_model

def train_model(model_args:EvalModelArguments):
    pretrained_model_name = model_args.eval_pretrained
    sts_num_epochs = model_args.epoch
    train_batch_size = model_args.batch_size

    data1 = da.Dataset.from_pandas(pd.read_csv(model_args.eval_train_path, sep='\\t'))
    data2 = da.Dataset.from_pandas(pd.read_csv(model_args.eval_test_path, sep='\\t'))

    sts_train_examples = make_sts_input_example(data1)
    sts_valid_examples = make_sts_input_example(data2)

    train_dataloader = DataLoader(
        sts_train_examples,
        shuffle=True,
        batch_size=train_batch_size,
    )

    # Load Embedding Model
    embedding_model = models.Transformer(
        model_name_or_path=pretrained_model_name,
        max_seq_length=256,
        do_lower_case=True
    )

    # Only use Mean Pooling -> Pooling all token embedding vectors of sentence.
    pooling_model = models.Pooling(
        embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )

    model = SentenceTransformer(modules=[embedding_model, pooling_model])

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )
    # Use CosineSimilarityLoss
    train_loss = losses.CosineSimilarityLoss(model=model)

    # warmup steps
    warmup_steps = math.ceil(len(sts_train_examples) * sts_num_epochs / train_batch_size * 0.1)
    logging.info("Warmup-steps: {}".format(warmup_steps))

    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        sts_valid_examples,
        name="sts-dev",
    )

    # Training
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=dev_evaluator,
        epochs=sts_num_epochs,
        evaluation_steps=1000,
        warmup_steps=warmup_steps,
        output_path=None
    )

    torch.save(model.state_dict(), model_args.eval_model_path)

    return model

def make_sts_input_example(dataset):
    input_examples = []
    for i, data in enumerate(dataset):
        sentence1 = data['sentence1']
        sentence2 = data['sentence2']
        score = data['score'] / 5.0  # normalize 0 to 5
        input_examples.append(InputExample(texts=[sentence1, sentence2], label=score))

    return input_examples