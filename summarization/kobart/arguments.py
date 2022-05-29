from typing import Optional
from dataclasses import dataclass, field

import yaml
from transformers import TrainingArguments, Seq2SeqTrainingArguments

def return_train_config():
    with open('./configs.yaml') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    training_args, model_args, data_args,wandb_args = configs['TrainingArguments'], \
                                                      configs['ModelArguments'], \
                                                      configs['DataTrainingArguments'], \
                                                      configs['WandbArguments']

    model_args = ModelArguments(**model_args)
    data_args = DataTrainingArguments(**data_args)
    training_args = Seq2SeqTrainingArguments(**training_args)
    wandb_args = WandbArguments(**wandb_args)

    return model_args, data_args, training_args, wandb_args

def return_eval_model_config():
    with open('./configs.yaml') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    eval_args = configs['EvalModelArguments']

    eval_args = EvalModelArguments(**eval_args)
    return eval_args

def return_inference_config():
    with open('./inference_configs.yaml') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    model_args = configs['ModelArguments']
    data_args = configs['DataTrainingArguments']
    gen_args = configs['GenerateArguments']

    model_args = ModelArguments(**model_args)
    data_args = DataTrainingArguments(**data_args)
    gen_args = GenerateArguments(**gen_args)

    return model_args, data_args, gen_args

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )
    use_checkpoint: bool = field(
        default=False,
        metadata={
            "help": (
                "If you want to use Checkpoint during training, set this value to True"
            )
        },
    )


@dataclass
class DataTrainingArguments:
    train_file: str = field(
        default=None, metadata={"help": "The directory of input training data file (a jsonlines or csv file)."}
    )
    validation_file: str = field(
        default=None,
        metadata={
            "help": (
                "An optional input directory of evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    saved_data_path: str = field(
        default='./a.pickle', metadata={"help": "File Path which is saved Train & Validation Data"}
    )

    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

@dataclass
class WandbArguments:
    project: str = field(
        metadata={"help": "Wandb Project Name"}
    )
    entity: str = field(
        metadata={"help": "Wandb Entity Name"}
    )
    name: str = field(
        metadata={"help" : "Train_Name"}
    )

class EvalModelArguments:
    eval_model_path: str = field(
        metadata={"help": "Path to pretrained SBERT Model"}
    )
    eval_pretrained: str = field(
        metadata={"help": "SBERT's pretrained Model(BERT Model Name)"}
    )
    eval_train_path: str = field(
        metadata = {"help": "SBERT's train data"}
    )
    eval_test_path: str = field(
        metadata = {"help": "SBERT's test data"}
    )
    batch_size: str = field(
        metadata = {"help": "SBERT train data batch size"}
    )
    epoch: str = field(
        metadata = {"help": "SBERT train EPOCH"}
    )

@dataclass
class GenerateArguments:
    num_beams: int = field(
        metadata= {"help" : "Number of Beams"}
    )

    max_length: int = field(
        metadata = {"help": "Generated Sentence Max Length"}
    )

    top_k: int=field(
        metadata = {"help": "Token with probability ranking outside top_k are excluded from sampling"}
    )

    top_p: float = field(
        metadata= {"help": "Create Only from a set of candidatees with {top_p * 100}% cummulative probabilities"}
    )
