import torch
from transformers import PreTrainedTokenizerFast, Seq2SeqTrainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, HPSearchBackend, ShardedDDPOption
from transformers.trainer_pt_utils import reissue_pt_warnings
from typing import Optional, Tuple
from pathlib import Path
import json
import os
import warnings
import numpy as np

# https://github.com/SKT-AI/KoBART/issues/14
class PTTFwithSaveVocab(PreTrainedTokenizerFast):
    def __init__ (self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        See https://huggingface.co/transformers/_modules/transformers/tokenization_utils_base.html#save_vocabulary for docstring of original save_vocabulary.
        """
        file_path = os.path.expanduser(save_directory)
        Path(file_path).mkdir(parents=True, exist_ok=True)
        prefix = filename_prefix if filename_prefix else ""
        full_path_with_filename = file_path + '/' + prefix + 'vocab.json' # 오타 난 부분
        with open(full_path_with_filename, 'w', encoding='utf-8') as vocab_file:
            json.dump(self.get_vocab(), vocab_file, ensure_ascii=False)
        
        return (full_path_with_filename,)