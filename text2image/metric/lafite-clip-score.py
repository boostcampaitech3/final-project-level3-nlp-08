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

# text2image generator 정의
class Generator:
    def __init__(self, device, path):
        self.name = 'generator'
        self.model = self.load_model(device, path)
        self.device = device
        self.force_32 = False
        
    def load_model(self, device, path):
        with dnnlib.util.open_url(path) as f:
            network= legacy.load_network_pkl(f)
            self.G_ema = network['G_ema'].to(device)
            self.D = network['D'].to(device)

            return self.G_ema
        
    def generate(self, z, c, fts, noise_mode='const', return_styles=True):
        return self.model(z, c, fts=fts, noise_mode=noise_mode, return_styles=return_styles, force_fp32=self.force_32)
    
    def generate_from_style(self, style, noise_mode='const'):
        ws = torch.randn(1, self.model.num_ws, 512)
        return self.model.synthesis(ws, fts=None, styles=style, noise_mode=noise_mode, force_fp32=self.force_32)
    
    def tensor_to_img(self, tensor):
        img = torch.clamp((tensor + 1.) * 127.5, 0., 255.)
        img_list = img.permute(0, 2, 3, 1)
        img_list = [img for img in img_list]
        return Image.fromarray(torch.cat(img_list, dim=-2).detach().cpu().numpy().astype(np.uint8))


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
    # lafite 실행 및 clip score 산출
    with torch.no_grad():

        device = 'cuda:0' # please use GPU, do not use CPU
        path = 'pre-trained-google-cc-best-fid.pkl'
        # path = './some_pre-trained_models.pkl'  # pre-trained model
        generator = Generator(device=device, path=path)
        clip_model, preprocess_clip = clip.load("ViT-B/32", device=device)
        clip_model = clip_model.eval()
        
        if args.num_images_to_generate > 1:
            tokenized_text = clip.tokenize([args.txt]*args.num_images_to_generate).to(device)
            txt_fts = clip_model.encode_text(tokenized_text)
            txt_fts = txt_fts/txt_fts.norm(dim=-1, keepdim=True)

            z = torch.randn((args.num_images_to_generate, 512)).to(device)
            c = torch.randn((args.num_images_to_generate, 1)).to(device) # label is actually not used
            img, _ = generator.generate(z=z, c=c, fts=txt_fts)
            to_show_img = generator.tensor_to_img(img)
            to_show_img.save('./generated.jpg')
        else:
            tokenized_text = clip.tokenize([args.txt]).to(device)
            txt_fts = clip_model.encode_text(tokenized_text)
            txt_fts = txt_fts/txt_fts.norm(dim=-1, keepdim=True)

            z = torch.randn((1, 512)).to(device)
            c = torch.randn((1, 1)).to(device) # label is actually not used
            img, _ = generator.generate(z=z, c=c, fts=txt_fts)
            to_show_img = generator.tensor_to_img(img)
            to_show_img.save('./generated.jpg')
        
        clip_model.to(device=device)
        scores, rank = clip_score_rank(prompt=args.txt,
                    images=[np.array(to_show_img)],
                    model_clip=clip_model,
                    preprocess_clip=preprocess_clip,
                    device=device)

    return scores


if __name__=="__main__":
    # 텍스트 입력 및 생성할 이미지 개수 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt", type=str, default="an armchair in the shape of an avocado")
    parser.add_argument("--num", dest="num_images_to_generate", type=int, default=1)
    args = parser.parse_args()
    main(args)