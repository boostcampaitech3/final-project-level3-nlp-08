from fastapi import FastAPI
from service.back_function import *
from pydantic import BaseModel

# from text2image.model.dalle.models import Rep_Dalle
from transformers import BartForConditionalGeneration, AutoTokenizer
# from text2image.model.dalle.utils.utils import set_seed, clip_score

import clip
import json
import numpy as np

# load model
global model
model = BartForConditionalGeneration.from_pretrained('chi0/kobart-dial-sum')
global tokenizer
tokenizer = AutoTokenizer.from_pretrained('chi0/kobart-dial-sum')
# global txt2imgModel
# txt2imgModel,_ = Rep_Dalle.from_pretrained("text2image/model/tf_model/model/29052022_082436")

app = FastAPI()

global client
with open("./client.json", "r") as file:
    client = json.load(file)

################ 요약 ################
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

################ 번역 + 전처리 ################
def ko2en(sentence):
    global client
    client_id, client_secret = client["client_id"], client["client_secret"]
    sentence = mt(sentence, client_id, client_secret)
    sentences = preprocess(sentence)
    return sentences

"""
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

    new_im = np.array(new_im)
    new_im = new_im.tolist()
    return new_im
"""

################ Text2Image ################

class Item(BaseModel):
    dialogue: str

@app.post('/upload')
async def upload_image(item: Item):
    kor_sum = postprocess_text_first_sent(generate_summary(item.dialogue))

    result = ko2en(kor_sum[0])
    # stop_img = txt2img(result[0])

    # return {"summary": result, "kor_sum":kor_sum[0], "image_array":stop_img}
    return {"summary": result, "kor_sum":kor_sum[0]}