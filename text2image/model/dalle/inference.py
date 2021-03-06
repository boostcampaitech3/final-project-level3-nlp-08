# ------------------------------------------------------------------------------------
# Modified from minDALL-E (https://github.com/kakaobrain/minDALL-E)
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
from translate import client, ko2en
import os
import sys
import argparse
import clip
import numpy as np
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dalle.models import Dalle, Rep_Dalle
from dalle.utils.utils import set_seed, clip_score


parser = argparse.ArgumentParser()
# 추가된 argument
parser.add_argument('--usetf', type=bool, default=True) # transfer learned model 사용 여부 : 사용(default)
parser.add_argument('--model_dir', type=str, default="../tf_model/tf_model/29052022_082436") # transfer learned model 경로
parser.add_argument('--client', type=str, default='./client.json') # papago api client 정보 저장된 json 위치 : 사전에 papago api발급 받아서 json파일로 정보 저장
# 기존 argument
parser.add_argument('-n', '--num_candidates', type=int, default=3)
parser.add_argument('--prompt', type=str, default='A painting of a tree on the ocean') # input sentence(str) / input list object/ input txt path
parser.add_argument('--softmax-temperature', type=float, default=1.0)
parser.add_argument('--top_k', type=int, default=256)
parser.add_argument('--top-p', type=float, default=None, help='0.0 <= top-p <= 1.0')
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()

# Setup
assert args.top_k <= 256, "It is recommended that top_k is set lower than 256."

set_seed(args.seed)
device = 'cuda:0'
model = None
if args.usetf:
    # Load transfer learned model
    model,_ = Rep_Dalle.from_pretrained(args.model_dir)  # This will automatically download the pretrained model.
else:
    # Load pretrained model
    model = Dalle.from_pretrained('minDALL-E/1.3B')  # This will automatically download the pretrained model.

model.to(device=device)


# Sampling : enTexts = [stopwords버전, tokNJR버전 문장 문장] ==> 문장 2개
enTexts = ko2en(args.prompt, args.client)

for text in enTexts:
    images = model.sampling(prompt=text,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            softmax_temperature=args.softmax_temperature,
                            num_candidates=args.num_candidates,
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

    # Print scores and save files
    print(f"원문:", args.prompt)
    print(f"번역문:", text)
    scores = sorted(scores, reverse=True)
    for i, score in enumerate(scores):
        print(i+1,"clip score:", score.item())

    # Save images
    images = images[ranks]
    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    for i in range(min(16, args.num_candidates)):
        im = Image.fromarray((images[i]*255).astype(np.uint8))
        im.save(f'./figures/{text}_{i}.png')