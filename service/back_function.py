import json
import urllib.request

from PIL import Image
import numpy as np
from text2image.model.dalle.utils.utils import set_seed, clip_score
import clip

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
    split_by_and = sentence.split(" and ")[0]
    sentence = word_tokenize(split_by_and)
    tags = pos_tag(sentence)
    for (word, tag) in tags:
        if tag[0]=='N' or tag[0]=='V' or tag[0]=='J' or tag[0]=='R':
            tokenized.append(word)

    return tokenized

def tokSTOP(sentence):
    sw = stopwords.words('english')
    split_by_and = sentence.split(" and ")[0]
    sentence = word_tokenize(split_by_and.lower())
    words = [word for word in sentence if word not in sw]
    
    return words

################ 번역 + 문장 변형(T2I) 과정 ################
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


def ko2en(client, sentence):
    client_id, client_secret = client["client_id"], client["client_secret"]
    sentence = mt(sentence, client_id, client_secret)
    sentences = preprocess(sentence)
    return sentences


################ Text-to-Image 함수 ################
def txt2img(txt2imgModel, text):
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

    new_im = np.array(new_im)
    new_im = new_im.tolist()
    return new_im