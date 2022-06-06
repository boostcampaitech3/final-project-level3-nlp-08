from pydantic import BaseModel

import json
import urllib.request

################ 전처리를 위한 Module 다운 ################
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

################ 후처리(Dialogue Summarization) ################
def postprocess_text_first_sent(preds):
    preds = [pred.strip() for pred in preds]
    preds = [pred[:pred.index(".")+1] if "." in pred else pred for pred in preds]

    return preds

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

################ 전처리(T2I) 함수 ################
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

################ 전처리(T2I) 과정 ################
def transformText(text):
    sentences = []
    sentences.append(", ".join(tokSTOP(text)))
    sentences.append(", ".join(tokNVJR(text)))
    return sentences


def preprocess(sentence):
    prefix = "A painting of "
    answer = []

    for sentence in transformText(sentence):
        answer.append(prefix + sentence)
    
    return answer