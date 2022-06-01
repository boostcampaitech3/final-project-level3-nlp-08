from typing import List
import urllib.request
import json

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

def cleanList(sentences:List):
    answers = []
    for sentence in sentences:
        answers.append(sentence[0])
    return answers