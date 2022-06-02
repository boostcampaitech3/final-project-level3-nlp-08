from typing import List
import urllib.request
import json

import nltk
from nltk.tokenize import word_tokenize
# from nltk.tag import pos_tag
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# !necessary : papago api client 정보를 저장한 json file 과 해당 file path


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

def tokSTOP(sentence):
    sw = stopwords.words('english')
    sentence = word_tokenize(sentence.lower())
    words = [word for word in sentence if word not in sw]
    
    return words


def transformText(text):
    words = tokSTOP(text)
    sentence = ", ".join(words)
    return sentence


def preprocess(sentence):
    prefix = "A painting of "
    text = transformText(sentence)
    answer = prefix + text
    
    return answer



################ 번역 + 전처리 ################

def ko2en(sentence, json_file):
    client_id, client_secret = client(json_file)
    sentence = mt(sentence, client_id, client_secret)
    sentences = preprocess(sentence)
    return sentences