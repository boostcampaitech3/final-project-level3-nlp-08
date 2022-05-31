# ------------------------------------------------------------------------------------
# minDALL-E
# Copyright (c) 2021 Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
from translate import client, cleanList, mt
import os
import sys
import argparse
import clip
import numpy as np
from PIL import Image
from translate import client, mt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dalle.models import Dalle, Rep_Dalle
from dalle.utils.utils import set_seed, clip_score


parser = argparse.ArgumentParser()
# 추가된 argument
parser.add_argument('--usetf', type=bool, default=False) # transfer learned model 사용 여부 : 사용(default)
parser.add_argument('--model_dir', type=str, default="../exp2_ep4/exp2_ep4/29052022_082436") # transfer learned model 경로
parser.add_argument('--client', type=int, default='./client.json') # papago api client 정보 저장된 json 위치 : 사전에 papago api발급 받아서 json파일로 정보 저장
parser.add_argument('--input_type', type=str, default="str")       # input type이 txt(txt파일 경로)인지, list(summ모델의 output형태)인지, str인지
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
    # model = torch.load('../model/pytorch_model.pth')
    # model,config = ImageGPT.from_pretrained("minDALL-E/1.3B",'../configs/exp1_ep4.yaml')
    # chck = torch.load("../exp2_ep4/exp2_ep4/29052022_082436/ckpt/last.ckpt")
    # model.load_state_dict(chck['state_dict'])
    # print(model)
else:
    # Load pretrained model
    model = Dalle.from_pretrained('minDALL-E/1.3B')  # This will automatically download the pretrained model.

model.to(device=device)

# Get client id & secret
client_id, client_secret = client(args.client)

# check if str or file path
if args.input_type == "str":
    # Sampling
    enText = mt(args.prompt, client_id, client_secret)
    images = model.sampling(prompt=enText,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            softmax_temperature=args.softmax_temperature,
                            num_candidates=args.num_candidates,
                            device=device).cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))

    # CLIP Re-ranking
    model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
    model_clip.to(device=device)
    ranks, scores = clip_score(prompt=enText,
                    images=images,
                    model_clip=model_clip,
                    preprocess_clip=preprocess_clip,
                    device=device)

    # Print scores and save files
    print(f"원문:", args.prompt)
    print(f"번역문:", enText)
    scores = sorted(scores, reverse=True)
    for i, score in enumerate(scores):
        print(i+1,"clip score:", score.item())

    # Save images
    images = images[ranks]
    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    for i in range(min(16, args.num_candidates)):
        im = Image.fromarray((images[i]*255).astype(np.uint8))
        im.save(f'./figures/{args.prompt}_{i}.png')
else:
    if args.inpu_type == "txt":
        sentences = None
        with open(args.prompt, "r") as f:
            sentences = f.realines()
    else:
        sentences = cleanList(args.prompt)  # summarization 모델의 output을 변형없이 그대로 가져올 때
    for idx, sentence in enumerate(sentences):
        enText = mt(sentence, client_id, client_secret)

        # Sampling
        images = model.sampling(prompt=enText,
                                top_k=args.top_k,
                                top_p=args.top_p,
                                softmax_temperature=args.softmax_temperature,
                                num_candidates=args.num_candidates,
                                device=device).cpu().numpy()
        images = np.transpose(images, (0, 2, 3, 1))

        # CLIP Re-ranking
        model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
        model_clip.to(device=device)
        ranks, scores = clip_score(prompt=enText,
                        images=images,
                        model_clip=model_clip,
                        preprocess_clip=preprocess_clip,
                        device=device)

        # Print scores and save files
        print(f"{idx} 번째 문장:")
        print("원문:", sentence)
        print("번역문:", enText)
        scores = sorted(scores, reverse=True)
        for i, score in enumerate(scores):
            print(i+1,"clip score:", score.item())

        # Save images
        txtname = args.dir.split(".")[1] # txt파일 이름
        images = images[ranks]

        if not os.path.exists('./figures'):
            os.makedirs('./figures')
        for i in range(min(16, args.num_candidates)):
            im = Image.fromarray((images[i]*255).astype(np.uint8))
            im.save(f'./figures/{txtname}/{idx}_{i}.png') # figures/txt_파일_이름/idx_i.png (txt파일 내 idx번째 문장 i위 이미지)