import os
import json
import pandas as pd
from pandas import json_normalize
from tqdm import tqdm
import pickle
import logging
import datasets.arrow_dataset as da

def return_data(logger:logging.Logger):
    if os.path.exists("./cache_data/raw_data.pickle"):
        logger.info('Get a Data that exists')
        with open("./cache_data/raw_data.pickle", "rb") as f:
            raw_datasets = pickle.load(f)

    else:
        sample_dataset = da.Dataset.from_pandas(get_raw_data(logger=logger))
        raw_datasets = sample_dataset.map(flatten, remove_columns=['id'], batched = True)

        with open("./cache_data/raw_data.pickle", "wb") as f:
            pickle.dump(raw_datasets, f)

    train_data_txt, validation_data_txt = raw_datasets.train_test_split(test_size=0.01).values()
    return train_data_txt, validation_data_txt

def flatten(example):
    dialogue_list = []

    for dict_data in example['dialogue']:
        return_string = ""
        for string in dict_data:
            return_string += string['participantID'] + ": " + string['utterance'] + "\r\n"

        dialogue_list.append(return_string[:-2])

    return {
        "dialogue": dialogue_list,
        "summary": example['summary']
    }

def get_raw_data(logger:logging.Logger):

    logger.info('Create a new DataFrame to use')
    train_json_data = []

    logger.info('Start to Read JSON File')
    for filename in tqdm(os.listdir("./data/Training")):
        with open(os.path.join("./data/Training", filename), 'r') as f:
            train_json_data.append(json.load(f))

    logger.info('End to Read JSON File')

    logger.info('Start JSON File to DataFrame')
    df = pd.concat([json_normalize(train_json_data[i]['data']) for i in tqdm(range(len(train_json_data)))])
    logger.info('End JSON File to DataFrame')

    dict_data = {'dialogue':df['body.dialogue'], 'summary':df['body.summary'], 'id':df['header.dialogueInfo.dialogueID']}

    return_df = pd.DataFrame(data=dict_data)

    return return_df

if __name__=='main':
    print(get_raw_data(logger=None).head())