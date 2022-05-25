import os
import numpy as np
import pandas as pd
from prometheus_client import Summary
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import argparse

class SummaryDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len, ignore_index=-100):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        # TODO: File 형식에 따라 변경 예정
        self.data = pd.read_csv(file_path)
        self.data_len = self.data.shape[0]

        self.pad_idx = self.tokenizer.pad_token_id
        self.ignore_index = ignore_index

    def pad_data(self, inputs):
        if len(inputs) < self.max_seq_len:
            pad = np.array([self.pad_idx] * (self.max_seq_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            # TODO: Truncation, overflow 처리하도록 수정 필요할듯
            inputs = inputs[:self.max_seq_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] * (self.max_seq_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            # TODO: Truncation, overflow 처리하도록 수정 필요할듯
            inputs = inputs[:self.max_seq_len]
        
        return inputs


    def __getitem__(self, idx):
        instance = self.data.iloc[idx]
        input_ids = self.tokenizer.encode(instance['dialogue'])
        input_ids = self.pad_data(input_ids)

        label_ids = self.tokenizer.encode(instance['summary'])
        label_ids.append(self.tokenizer.eos_token_id)
        dec_input_ids = [self.tokenizer.eos_token_id]   # bos_token_id 아닌가?
        label_ids = self.add_ignored_data(label_ids)

        return {'input_ids': np.array(input_ids, dtype=np.int_),
                'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                'lables': np.array(label_ids, dtype=np.int_)}
    
    def __len__(self):
        return self.len

class SummaryModule(pl.LightningDataModule):
    def __init__(self, train_file_path, test_file_path, tokenizer, max_seq_len=1026, batch_size=8, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.tokenizer = tokenizer
        self.num_workers = num_workers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers',
                            type=int,
                            default=4,
                            help='num of worker for dataloader')
        return parser

    def setup(self, stage):
        self.trainset = SummaryDataset(self.train_file_path, self.tokenizer, self.max_seq_len)
        self.testset = SummaryDataset(self.test_file_path, self.tokenizer, self.max_seq_len)

    # TODO: dataloader부분 수정 필요
    def train_dataloader(self):
        train = DataLoader(self.trainset,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers, shuffle=True)
        return train
    
    # TODO: validation dataloader 수정 필요
    def val_dataloader(self):
        val = DataLoader(self.testset,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers, shuffle=False)
        return val

    def test_dataloader(self):
        test = DataLoader(self.testset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)
        return test