import pandas as pd
import datasets.arrow_dataset as da
import math
import logging
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample

def make_sts_input_example(dataset):
    input_examples = []
    for i, data in enumerate(dataset):
        sentence1 = data['sentence1']
        sentence2 = data['sentence2']
        score = data['score'] / 5.0  # normalize 0 to 5
        input_examples.append(InputExample(texts=[sentence1, sentence2], label=score))

    return input_examples

def return_metric_model():
    pretrained_model_name = 'klue/roberta-base'
    sts_num_epochs = 4
    train_batch_size = 32

    sts_model_save_path = 'output/training_sts-' \
                          + pretrained_model_name.replace("/","-") + '-'\
                          + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    data1 = da.Dataset.from_pandas(pd.read_csv('./data/KorSTS/sts-train.tsv', sep='\\t'))
    data2 = da.Dataset.from_pandas(pd.read_csv('./data/KorSTS/sts-dev.tsv', sep='\\t'))

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
        output_path=sts_model_save_path
    )

    torch.save(model.state_dict(), './cache_data/RDASS.pt')

    return model