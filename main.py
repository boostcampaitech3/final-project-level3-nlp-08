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
        max_length=128,
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
                             )

    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return output_str

def mt(sentence):
    global client
    client_id = client["client_id"]
    client_secret = client["client_secret"]
    
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

class Item(BaseModel):
    dialogue: str

@app.post('/upload')
async def upload_image(item: Item):
    kor_sum = postprocess_text_first_sent(generate_summary(item.dialogue))
    
    result = mt(kor_sum[0])

    return {"summary": result}