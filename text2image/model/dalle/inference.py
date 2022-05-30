# ------------------------------------------------------------------------------------
# minDALL-E
# Copyright (c) 2021 Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
import torch
import os
import sys
import argparse
import clip
import numpy as np
from PIL import Image
from translate import client, process_answers, mt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dalle.models import Dalle
from dalle.utils.utils import set_seed, clip_score


parser = argparse.ArgumentParser()
parser.add_argument('--usetf', type=bool, default=False) # transfer learned model 사용 여부
parser.add_argument('--model_dir', type=str, default="./model.ckpt") # transfer learned model 경로
parser.add_argument('--client', type=int, default='./client.json') # papago api client 정보 저장된 json 위치
parser.add_argument('-n', '--num_candidates', type=int, default=5)
parser.add_argument('--isfile', type=bool, default=False)          # input이 여러 문장이 "\n"를 기준으로 저장된 txt 파일 경로인지, 단일 문장 str인지
parser.add_argument('--txt', type=str, default='./sentences.txt')  # input korean txt file path
parser.add_argument('--prompt', type=str, default='A painting of a tree on the ocean') # input sentence
parser.add_argument('--softmax-temperature', type=float, default=1.0)
parser.add_argument('--top-k', type=int, default=256)
parser.add_argument('--top-p', type=float, default=None, help='0.0 <= top-p <= 1.0')
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()

# Setup
assert args.top_k <= 256, "It is recommended that top_k is set lower than 256."

set_seed(args.seed)
device = 'cuda:0'
# Load pretrained model
model = Dalle.from_pretrained('minDALL-E/1.3B')  # This will automatically download the pretrained model.
if args.usetf:
    # Load transfer learned model
    model.load_state_dict(torch.load(args.model_dir))

model.to(device=device)

# Get client id & secret
client_id, client_secret = client(args.client)

# check if str or file path
if args.isfile:
    # sentences = process_answers(output_list) # 대화요약 모델의 output list 그대로 가져올 때
    sentences = args.txt.realines()
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
        scores, ranks = clip_score(prompt=enText,
                        images=images,
                        model_clip=model_clip,
                        preprocess_clip=preprocess_clip,
                        device=device)

        # Prepare for scoring and save files
        print(f"{idx} 번째 문장:")
        print("원문:", sentence)
        print("번역문:", enText)
        for i, score in enumerate(scores):
            print(i+1,"clip score:", score)

        # Save images
        txtname = args.dir.split(".")[1] # txt파일 이름
        images = images[ranks]

        if not os.path.exists('./figures'):
            os.makedirs('./figures')
        for i in range(min(16, args.num_candidates)):
            im = Image.fromarray((images[i]*255).astype(np.uint8))
            im.save(f'./figures/{txtname}/{idx}_{i}.png')

else:
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
    scores, ranks = clip_score(prompt=enText,
                    images=images,
                    model_clip=model_clip,
                    preprocess_clip=preprocess_clip,
                    device=device)

    # Prepare for scoring and save files
    print(f"원문:", args.prompt)
    print(f"번역문:", enText)
    for i, score in enumerate(scores):
        print(i+1,"clip score:", score)

    # Save images
    images = images[ranks]
    print(ranks, images.shape)
    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    for i in range(min(16, args.num_candidates)):
        im = Image.fromarray((images[i]*255).astype(np.uint8))
        im.save(f'./figures/{args.prompt}_{i}.png')