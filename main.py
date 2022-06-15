from fastapi import FastAPI
from service.back_function import *
from pydantic import BaseModel

from service.image_to_text import Rep_Dalle
from transformers import BartForConditionalGeneration, AutoTokenizer

from fastapi import FastAPI
from service.back_function import *
from pydantic import BaseModel

from service.text_to_image import Rep_Dalle
from transformers import BartForConditionalGeneration, AutoTokenizer

import json

global client
with open("./client.json", "r") as file:
    client = json.load(file)

# load model
global model
model = BartForConditionalGeneration.from_pretrained('chi0/kobart-dial-sum')
global tokenizer
tokenizer = AutoTokenizer.from_pretrained('chi0/kobart-dial-sum')
global txt2imgModel
txt2imgModel,_ = Rep_Dalle.from_pretrained("service/29052022_082436")


app = FastAPI()
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


class Item(BaseModel):
    dialogue: str

@app.post('/upload')
async def upload_image(item: Item):
    kor_sum = postprocess_text_first_sent(generate_summary(item.dialogue))

    return {"kor_sum":kor_sum[0]}

@app.post('/images')
async def make_image(item: Item):
    result = ko2en(client, item.dialogue)
    stop_img = txt2img(txt2imgModel, result[0])

    return {"image_array":stop_img}
