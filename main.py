from typing import Optional, List
from fastapi import FastAPI, File, UploadFile
import os
import streamlit as st

import io
import os
import yaml
import json

import urllib.request
import json

from fastapi import FastAPI
from pydantic import BaseModel

from googletrans import Translator

from transformers import BartForConditionalGeneration, AutoTokenizer

import googletrans

import urllib.request
import json

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# !necessary : papago api client 정보를 저장한 json file 과 해당 file path


global model
model = BartForConditionalGeneration.from_pretrained('chi0/kobart-dial-sum')
global tokenizer
tokenizer = AutoTokenizer.from_pretrained('chi0/kobart-dial-sum')

app = FastAPI()

global client
with open("./client.json", "r") as file:
    client = json.load(file)


def postprocess_text_first_sent(preds):
    preds = [pred.strip() for pred in preds]
    preds = [pred[:pred.index(".")+1] if "." in pred else pred for pred in preds]

    return preds

def generate_summary(dialogue:str):
    global model
    global tokenizer

    inputs = tokenizer(
        dialogue,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    outputs = model.generate(input_ids,
                             num_beams=5,
                             max_length=64,
                             attention_mask=attention_mask,
                             top_k=50,
                             top_p=0.95,
                             no_repeat_ngram_size=3,
                             temperature=0.7
                             )

    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return output_str


################ 번역 ################

def client(json_file):
    with open(json_file, "r") as file:
        client = json.load(file)

    client_id = client["client_id"]
    client_secret = client["client_secret"]

    return client_id, client_secret


def mt(sentence, client_id, client_secret):
    koText = urllib.parse.quote(sentence)
    data = "source=ko&target=en&text=" + koText
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        json_result = json.loads(response_body.decode('utf-8'))
        enText = json_result["message"]["result"]["translatedText"]
        return enText
    else:
        print("Error Code:" + rescode)


def ko2en(sentence, client_id, client_secret):
    sentence = preprocess(sentence)
    return mt(sentence, client_id, client_secret)


################ 전처리 ################

def tokNJR(sentence):
            tokenized = []
            sentence = word_tokenize(sentence)
            tags = pos_tag(sentence)
            for (word, tag) in tags:
                if tag[0]=='N' or tag[0]=='J' or tag[0]=='R':
                    tokenized.append(word)

            return tokenized


def tokSTOP(sentence):
    sw = stopwords.words('english')
    sentence = word_tokenize(sentence.lower())
    words = [word for word in sentence if word not in sw]
    
    return words


def transformText(text):

    sentences = []
    sentences.append(", ".join(tokSTOP(text)))
    sentences.append(", ".join(tokNJR(text)))
    return sentences


def preprocess(sentence):
    prefix = "A painting of "
    answer = []
    for sentence in transformText(sentence):
        answer.append(prefix + sentence)
    
    return answer

################ 번역 + 전처리 ################

def ko2en(sentence, json_file):
    client_id, client_secret = client(json_file)
    sentence = mt(sentence, client_id, client_secret)
    sentences = preprocess(sentence)
    return sentences


class Item(BaseModel):
    dialogue: str

@app.post('/upload')
async def upload_image(item: Item):
    kor_sum = postprocess_text_first_sent(generate_summary(item.dialogue))
    
    result = mt(kor_sum[0])

    return {"summary": result}