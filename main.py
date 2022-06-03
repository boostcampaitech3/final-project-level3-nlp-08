from typing import Optional, List
from fastapi import FastAPI, File, UploadFile
import os
import streamlit as st
import subprocess

import io
import os
import yaml
import json

import urllib.request

from pydantic import BaseModel
<<<<<<< HEAD

# from googletrans import Translator
from text2image.model.dalle.models import Rep_Dalle
from text2image.model.dalle.models import Dalle
from transformers import BartForConditionalGeneration, AutoTokenizer

# import googletrans
=======
from transformers import BartForConditionalGeneration, AutoTokenizer
from text2image.model.dalle.models import Rep_Dalle
from transformers import BartForConditionalGeneration, AutoTokenizer

from text2image.model.dalle.utils.utils import set_seed, clip_score
import clip
from PIL import Image
import numpy as np
>>>>>>> 36262fb8a5764d2b40554d800131bb5346d54ea0

import urllib.request
import json

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
# !necessary : papago api client 정보를 저장한 json file 과 해당 file path

<<<<<<< HEAD
from text2image.model.dalle.utils.utils import set_seed, clip_score
import clip
from PIL import Image
import numpy as np

=======
# load model
>>>>>>> 36262fb8a5764d2b40554d800131bb5346d54ea0
global model
model = BartForConditionalGeneration.from_pretrained('chi0/kobart-dial-sum')
global tokenizer
tokenizer = AutoTokenizer.from_pretrained('chi0/kobart-dial-sum')
global txt2imgModel
txt2imgModel,_ = Rep_Dalle.from_pretrained("text2image/model/tf_model/model/29052022_082436")

########### txt2img ###########
global txt2imgModel
txt2imgModel = Dalle.from_pretrained('minDALL-E/1.3B')
########### txt2img ###########

app = FastAPI()

global client
with open("./text2image/model/dalle/client.json", "r") as file:
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


################ 전처리 ################


def tokNVJR(sentence):
    tokenized = []
    sentence = word_tokenize(sentence)
    tags = pos_tag(sentence)
    for (word, tag) in tags:
        if tag[0]=='N' or tag[0]=='V' or tag[0]=='J' or tag[0]=='R':
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
    sentences.append(", ".join(tokNVJR(text)))
    return sentences


def preprocess(sentence):
    prefix = "A painting of "
    answer = []
    print(transformText(sentence))
    for sentence in transformText(sentence):
        answer.append(prefix + sentence)
    
    return answer

################ 번역 + 전처리 ################

def ko2en(sentence):
    global client
    client_id, client_secret = client["client_id"], client["client_secret"]
    sentence = mt(sentence, client_id, client_secret)
    sentences = preprocess(sentence)
    return sentences

################ Text2Image ################

def txt2img(text):
    set_seed(42)
    device = 'cuda:0'
    txt2imgModel.to(device=device)

    # Sampling : enTexts = [stopwords버전, tokNJR버전 문장 문장] ==> 문장 2개
    images = txt2imgModel.sampling(prompt=text,
                            top_k=256,
                            top_p=None,
                            softmax_temperature=1.0,
                            num_candidates=3,
                            device=device).cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))

    # CLIP Re-ranking
    model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
    model_clip.to(device=device)
    ranks, scores = clip_score(prompt=text,
                    images=images,
                    model_clip=model_clip,
                    preprocess_clip=preprocess_clip,
                    device=device)

    # Save images
    images = images[ranks]
    im = []
    for i in range(3):
        im.append(Image.fromarray((images[i]*255).astype(np.uint8)))

    widths, heights = zip(*(i.size for i in im))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for i in im:
        new_im.paste(i, (x_offset,0))
        x_offset += i.size[0]

<<<<<<< HEAD
    return np.array(new_im)
=======
    new_im = np.array(new_im)
    new_im = new_im.tolist()
    return new_im
>>>>>>> 36262fb8a5764d2b40554d800131bb5346d54ea0

################ Text2Image ################

class Item(BaseModel):
    dialogue: str

@app.post('/upload')
async def upload_image(item: Item):
    kor_sum = postprocess_text_first_sent(generate_summary(item.dialogue))

    result = ko2en(kor_sum[0])
    stop_img = txt2img(result[0])
    # NVJR_imgs = txt2img(result[1])
<<<<<<< HEAD

    return {"image_array":stop_img}
    # return {"summary": result}
=======


    return {"summary": result, "kor_sum":kor_sum[0], "image_array":stop_img}
>>>>>>> 36262fb8a5764d2b40554d800131bb5346d54ea0
