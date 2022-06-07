# 라이브러리 불러오기
import torch
import numpy as np
import pickle
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import dnnlib, legacy
import clip
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm
import scipy
import argparse


# clip score & rank function 정의
@torch.no_grad()
def clip_score_rank(prompt: str,
               images: np.ndarray,
               model_clip: torch.nn.Module,
               preprocess_clip,
               device: str) -> np.ndarray:
    images = [preprocess_clip(Image.fromarray((image*255).astype(np.uint8))) for image in images]
    images = torch.stack(images, dim=0).to(device=device)
    texts = clip.tokenize(prompt).to(device=device)
    texts = torch.repeat_interleave(texts, images.shape[0], dim=0)

    image_features = model_clip.encode_image(images)
    text_features = model_clip.encode_text(texts)

    scores = F.cosine_similarity(image_features, text_features).squeeze()
    rank = torch.argsort(scores, descending=True).cpu().numpy()
    return scores, rank


def main(args):
    # image -> numpy array 형변환
    img = Image.open(args.img_path)
    img = np.array(img)
    
    # clip score 산출
    device = 'cuda:0' # please use GPU, do not use CPU
    clip_model, preprocess_clip = clip.load("ViT-B/32", device=device)        
    
    clip_model.to(device=device)
    scores, rank = clip_score_rank(prompt=args.txt,
                images=[img],
                model_clip=clip_model,
                preprocess_clip=preprocess_clip,
                device=device)

    return sorted(scores, reverse=True)


if __name__=="__main__":
    # 텍스트 입력 및 생성한 이미지 경로 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt", type=str, default="A painting of a tree on the ocean")
    parser.add_argument("--path", dest="img_path", type=str, default="../assets/example.png")
    args = parser.parse_args()
    main(args)