import os
import json
import pandas as pd
from pandas import json_normalize
from tqdm import tqdm
import pickle
import logging


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

    if os.path.exists("./cache_data/raw_data.pickle"):
        logger.info('Get a DataFrame that exists')
        with open("./cache_data/raw_data.pickle", "rb") as f:
            return_df = pickle.load(f)
    else:
        logger.info('Create a new DataFrame to use')
        train_json_data = []

        logger.info('Start to Read JSON File')
        for filename in tqdm(os.listdir("./data/Training")):
            with open(os.path.join("./data/Training", filename), 'r') as f:
                train_json_data.append(json.load(f))
        logger.info('End to Read JSON File')

        """
        for filename in os.listdir("../data/Validation"):
            with open(os.path.join("../data/Validation", filename), 'r') as f:
                train_json_data.append(json.load(f))
        """

        logger.info('Start JSON File to DataFrame')
        df = pd.concat([json_normalize(train_json_data[i]['data']) for i in tqdm(range(len(train_json_data)))])
        logger.info('End JSON File to DataFrame')

        dict_data = {'dialogue':df['body.dialogue'], 'summary':df['body.summary'], 'id':df['header.dialogueInfo.dialogueID']}

        return_df = pd.DataFrame(data=dict_data)

        with open("./cache_data/raw_data.pickle", "wb") as f:
            pickle.dump(return_df, f)

    return return_df

if __name__=='main':
    print(get_raw_data(logger=None).head())