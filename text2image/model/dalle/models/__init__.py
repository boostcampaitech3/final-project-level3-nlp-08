# ------------------------------------------------------------------------------------
# minDALL-E
# Copyright (c) 2021 Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Tuple
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import functional as F
from .stage1.vqgan import VQGAN
from .stage2.transformer import Transformer1d, iGPT
from .. import utils
from ..utils.config import get_base_config
from ..utils.sampling import sampling, sampling_igpt, get_positional_encoding
from .tokenizer import build_tokenizer

_MODELS = {
    'minDALL-E/1.3B': 'https://arena.kakaocdn.net/brainrepo/models/minDALL-E/57b008f02ceaa02b779c8b7463143315/1.3B.tar.gz'
}
USE_CUDA = torch.cuda.is_available()
device = 'cpu'
if USE_CUDA:
    device = 'cuda:0'


class Dalle(nn.Module):
    def __init__(self,
                 config: OmegaConf) -> None:
        super().__init__()
        self.tokenizer = None
        self.stage1 = VQGAN(n_embed=config.stage1.n_embed,
                            embed_dim=config.stage1.embed_dim,
                            hparams=config.stage1.hparams)
        self.stage2 = Transformer1d(vocab_size_txt=config.stage2.vocab_size_txt,
                                    vocab_size_img=config.stage2.vocab_size_img,
                                    hparams=config.stage2.hparams)
        self.config_stage1 = config.stage1
        self.config_stage2 = config.stage2
        self.config_dataset = config.dataset

    @classmethod
    def from_pretrained(cls,
                        path: str) -> nn.Module:
        path = _MODELS[path] if path in _MODELS else path
        path = utils.realpath_url_or_path(path, root=os.path.expanduser("~/.cache/minDALL-E"))

        config_base = get_base_config()
        config_new = OmegaConf.load(os.path.join(path, 'config.yaml'))
        config_update = OmegaConf.merge(config_base, config_new)

        model = cls(config_update)
        model.tokenizer = build_tokenizer(os.path.join(path, 'tokenizer'),
                                          context_length=model.config_dataset.context_length,
                                          lowercase=True,
                                          dropout=None)
        model.stage1.from_ckpt(os.path.join(path, 'stage1_last.ckpt'))
        model.stage2.from_ckpt(os.path.join(path, 'stage2_last.ckpt'))
        return model

    @torch.no_grad()
    def sampling(self,
                 prompt: str,
                 top_k: int = 256,
                 top_p: Optional[float] = None,
                 softmax_temperature: float = 1.0,
                 num_candidates: int = 96,
                 device: str = device,
                 use_fp16: bool = True) -> torch.FloatTensor:
        self.stage1.eval()
        self.stage2.eval()

        tokens = self.tokenizer.encode(prompt)
        tokens = torch.LongTensor(tokens.ids)
        tokens = torch.repeat_interleave(tokens.unsqueeze(0), num_candidates, dim=0)

        # Check if the encoding works as intended
        # print(self.tokenizer.decode_batch(tokens.tolist(), skip_special_tokens=True)[0])

        tokens = tokens.to(device)
        codes = sampling(self.stage2,
                         tokens,
                         top_k=top_k,
                         top_p=top_p,
                         softmax_temperature=softmax_temperature,
                         use_fp16=use_fp16)
        codes = codes.view(num_candidates, 16, 16)  # [B, 16, 16]
        pixels = torch.clamp(self.stage1.decode_code(codes) * 0.5 + 0.5, 0, 1)  # [B, 256, 256]
        return pixels

class Rep_Dalle(pl.LightningModule):
    def __init__(self,
                 config: OmegaConf) -> None:
        super().__init__()
        self.tokenizer = None
        self.stage1 = VQGAN(n_embed=config.stage1.n_embed,
                            embed_dim=config.stage1.embed_dim,
                            hparams=config.stage1.hparams)
        self.stage2 = Transformer1d(vocab_size_txt=config.stage2.vocab_size_txt,
                                    vocab_size_img=config.stage2.vocab_size_img,
                                    hparams=config.stage2.hparams)
        self.config_stage1 = config.stage1
        self.config_stage2 = config.stage2
        self.config = config
        self.config_dataset = config.dataset

        
        self.stage1.eval()
        for p in self.stage1.parameters():
            p.requires_grad = False

    @classmethod
    def from_pretrained(cls,
                        path_upstream: str,
                        path_downstream: Optional[str] = None) -> Tuple[nn.Module, OmegaConf]:
        path_upstream = _MODELS[path_upstream] if path_upstream in _MODELS else path_upstream
        path_upstream = utils.realpath_url_or_path(path_upstream, root=os.path.expanduser("~/.cache/minDALL-E"))

        if path_downstream:
            config_base = get_base_config(use_default=False)
            config_down = OmegaConf.load(path_downstream)
            config_down = OmegaConf.merge(config_base, config_down)
        else:
            config_base = get_base_config()
            config_new = OmegaConf.load(os.path.join(path_upstream, 'config.yaml'))
            config_down = OmegaConf.merge(config_base, config_new)
        
        model = cls(config_down)
        
        model.tokenizer = build_tokenizer(os.path.join(path_upstream, 'tokenizer'),
                                          context_length=model.config_dataset.context_length,
                                          lowercase=True,
                                          dropout=None)
        if os.path.exists(os.path.join(path_upstream, 'stage1_last.ckpt')):
            model.stage1.from_ckpt(os.path.join(path_upstream, 'stage1_last.ckpt'), strict=True)
            model.stage2.from_ckpt(os.path.join(path_upstream, 'stage2_last.ckpt'), strict=False)
        elif os.path.exists(os.path.join(path_upstream, 'ckpt/last.ckpt')):
            chck = torch.load(os.path.join(path_upstream, 'ckpt/last.ckpt'),map_location=torch.device(device))
            model.load_state_dict(chck['state_dict'])
        return model, config_down

    @torch.no_grad()
    def sampling(self,
                 prompt: str,
                 top_k: int = 256,
                 top_p: Optional[float] = None,
                 softmax_temperature: float = 1.0,
                 num_candidates: int = 96,
                 device: str = device,
                 use_fp16: bool = True) -> torch.FloatTensor:
        self.stage1.eval()
        self.stage2.eval()

        tokens = self.tokenizer.encode(prompt)
        tokens = torch.LongTensor(tokens.ids)
        tokens = torch.repeat_interleave(tokens.unsqueeze(0), num_candidates, dim=0)

        # Check if the encoding works as intended
        # print(self.tokenizer.decode_batch(tokens.tolist(), skip_special_tokens=True)[0])

        tokens = tokens.to(device)
        codes = sampling(self.stage2,
                         tokens,
                         top_k=top_k,
                         top_p=top_p,
                         softmax_temperature=softmax_temperature,
                         use_fp16=use_fp16)
        codes = codes.view(num_candidates, 16, 16)  # [B, 16, 16]
        pixels = torch.clamp(self.stage1.decode_code(codes) * 0.5 + 0.5, 0, 1)  # [B, 256, 256]
        return pixels
    
    def sample(self,
                 prompt: str,
                 top_k: int = 256,
                 top_p: Optional[float] = None,
                 softmax_temperature: float = 1.0,
                 num_candidates: int = 96,
                 device: str = device,
                 use_fp16: bool = True) -> torch.FloatTensor:

        tokens = self.tokenizer.encode(prompt)
        tokens = torch.LongTensor(tokens.ids)
        tokens = torch.repeat_interleave(tokens.unsqueeze(0), num_candidates, dim=0)

        # Check if the encoding works as intended
        # print(self.tokenizer.decode_batch(tokens.tolist(), skip_special_tokens=True)[0])

        tokens = tokens.to(device)
        codes = sampling(self.stage2,
                         tokens,
                         top_k=top_k,
                         top_p=top_p,
                         softmax_temperature=softmax_temperature,
                         use_fp16=use_fp16)
        codes = codes.view(num_candidates, 16, 16)  # [B, 16, 16]
        pixels = torch.clamp(self.stage1.decode_code(codes) * 0.5 + 0.5, 0, 1)  # [B, 256, 256]
        return pixels, logits
    
    def forward(self,
                images: torch.FloatTensor,
                tokens: torch.LongTensor,) -> torch.FloatTensor:

        with torch.no_grad():
            with autocast(enabled=False):
                codes = self.stage1.get_codes(images).detach()
                
        codes_ = codes.clone().detach().to(device)
        pos_images = get_positional_encoding(codes_, mode='1d')
        
        pos_texts = get_positional_encoding(tokens, mode='1d')
        
        
        logits_img, logits_txt = self.stage2(codes, tokens.to(device), pos_images.to(device), pos_texts.to(device))
        return logits_img, logits_txt, codes
    
    def training_step(self, batch, batch_idx):
        images, texts = batch
        tokens = []
        for text in texts:
            tokens.append(self.tokenizer.encode(text).ids)
        # tokens = self.tokenizer.encode(texts)
        tokens = torch.LongTensor(tokens)
        logits_img, logits_txt, codes = self(images, tokens)
        loss = F.cross_entropy(logits_img.view(-1, logits_img.shape[-1]), codes.view(-1))
        # txt_loss = F.cross_entropy(logits_txt.view(-1, logits_txt.shape[-1]), codes.view(-1))
        # loss = img_loss + txt_loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, texts = batch
        tokens = []
        for text in texts:
            tokens.append(self.tokenizer.encode(text).ids)
        # tokens = self.tokenizer.encode(texts)
        tokens = torch.LongTensor(tokens)
        
        logits_img, logits_txt, codes = self(images, tokens)
        print(codes.shape)
        print(logits_img.shape)
        print(logits_txt.shape)
        loss = F.cross_entropy(logits_img.view(-1, logits_img.shape[-1]), codes.view(-1))
        # txt_loss = F.cross_entropy(logits_txt.view(-1, codes.shape[-1]), codes.view(-1))
        # loss = img_loss + txt_loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def configure_optimizers(self):
        assert self.config.optimizer.opt_type == 'adamW'
        assert self.config.optimizer.sched_type == 'cosine'

        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.config.optimizer.base_lr,
                                betas=self.config.optimizer.betas,
                                weight_decay=self.config.optimizer.weight_decay)
        sched = CosineAnnealingLR(opt,
                                  T_max=self.config.optimizer.max_steps,
                                  eta_min=self.config.optimizer.min_lr)
        sched = {
            'scheduler': sched,
            'name': 'cosine'
        }
        return [opt], [sched]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
                       on_tpu=False, using_native_amp=False, using_lbfgs=False):
        optimizer.step(closure=optimizer_closure)
        self.lr_schedulers().step()
        self.log("lr", self.lr_schedulers().get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def on_epoch_start(self):
        self.stage1.eval()

class ImageGPT(pl.LightningModule):
    def __init__(self,
                 config: OmegaConf) -> None:
        super().__init__()
        self.tokenizer = None
        self.stage1 = VQGAN(n_embed=config.stage1.n_embed,
                            embed_dim=config.stage1.embed_dim,
                            hparams=config.stage1.hparams)
        self.stage2 = iGPT(vocab_size_img=config.stage2.vocab_size_img,
                           use_cls_cond=config.stage2.use_cls_cond,
                           hparams=config.stage2.hparams)
        self.config = config
        self.config_dataset = config.dataset
        self.use_cls_cond = config.stage2.use_cls_cond

        # make the parameters in stage 1 not trainable
        self.stage1.eval()
        for p in self.stage1.parameters():
            p.requires_grad = False

    @classmethod
    def from_pretrained(cls,
                        path_upstream: str,
                        path_downstream: str) -> Tuple[nn.Module, OmegaConf]:
        path_upstream = _MODELS[path_upstream] if path_upstream in _MODELS else path_upstream
        path_upstream = utils.realpath_url_or_path(path_upstream, root=os.path.expanduser("~/.cache/minDALL-E"))
        config_base = get_base_config(use_default=False)
        config_down = OmegaConf.load(path_downstream)
        config_down = OmegaConf.merge(config_base, config_down)

        model = cls(config_down)
        
        model.tokenizer = build_tokenizer(os.path.join(path_upstream, 'tokenizer'),
                                          context_length=model.config_dataset.context_length,
                                          lowercase=True,
                                          dropout=None)
        
        model.stage1.from_ckpt(os.path.join(path_upstream, 'stage1_last.ckpt'), strict=True)
        model.stage2.from_ckpt(os.path.join(path_upstream, 'stage2_last.ckpt'), strict=False)
        return model, config_down

    @torch.no_grad()
    def sampling(self,
                 cls_idx: Optional[int] = None,
                 top_k: int = 256,
                 top_p: Optional[float] = None,
                 softmax_temperature: float = 1.0,
                 num_candidates: int = 96,
                 device: str = 'cuda:0',
                 use_fp16: bool = True,
                 is_tqdm: bool = True) -> torch.FloatTensor:
        self.stage1.eval()
        self.stage2.eval()

        if cls_idx is None:
            sos = self.stage2.sos.repeat(num_candidates, 1, 1)
        else:
            sos = torch.LongTensor([cls_idx]).to(device=device)
            sos = sos.repeat(num_candidates)
            sos = self.stage2.sos(sos).unsqueeze(1)

        codes = sampling_igpt(self.stage2,
                              sos=sos,
                              top_k=top_k,
                              top_p=top_p,
                              softmax_temperature=softmax_temperature,
                              use_fp16=use_fp16,
                              is_tqdm=is_tqdm)
        codes = codes.view(num_candidates, 16, 16)  # [B, 16, 16]
        pixels = torch.clamp(self.stage1.decode_code(codes) * 0.5 + 0.5, 0, 1)  # [B, 256, 256]
        return pixels

    def sample(self,
               cls_idx: Optional[int] = None,
               top_k: int = 256,
               top_p: Optional[float] = None,
               softmax_temperature: float = 1.0,
               num_candidates: int = 16,
               device: str = 'cuda:0',
               use_fp16: bool = True,
               is_tqdm: bool = True) -> torch.FloatTensor:
        self.stage1.eval()
        self.stage2.eval()

        if cls_idx is None:
            sos = self.stage2.sos.repeat(num_candidates, 1, 1)
        else:
            sos = torch.LongTensor([cls_idx]).to(device=device)
            sos = sos.repeat(num_candidates)
            sos = self.stage2.sos(sos).unsqueeze(1)

        codes = sampling_igpt(self.stage2,
                              sos=sos,
                              top_k=top_k,
                              top_p=top_p,
                              softmax_temperature=softmax_temperature,
                              use_fp16=use_fp16,
                              is_tqdm=is_tqdm)
        codes = codes.view(num_candidates, 16, 16)  # [B, 16, 16]
        pixels = torch.clamp(self.stage1.decode_code(codes) * 0.5 + 0.5, 0, 1)  # [B, 256, 256]
        return pixels

    def forward(self,
                images: torch.FloatTensor,
                labels: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        B, C, H, W = images.shape
        with torch.no_grad():
            with autocast(enabled=False):
                codes = self.stage1.get_codes(images).detach()
        logits = self.stage2(codes, labels)
        return logits, codes

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits, codes = self(images, labels=labels if self.use_cls_cond else None)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), codes.view(-1))
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits, codes = self(images, labels=labels if self.use_cls_cond else None)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), codes.view(-1))
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def configure_optimizers(self):
        assert self.config.optimizer.opt_type == 'adamW'
        assert self.config.optimizer.sched_type == 'cosine'

        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.config.optimizer.base_lr,
                                betas=self.config.optimizer.betas,
                                weight_decay=self.config.optimizer.weight_decay)
        sched = CosineAnnealingLR(opt,
                                  T_max=self.config.optimizer.max_steps,
                                  eta_min=self.config.optimizer.min_lr)
        sched = {
            'scheduler': sched,
            'name': 'cosine'
        }
        return [opt], [sched]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
                       on_tpu=False, using_native_amp=False, using_lbfgs=False):
        optimizer.step(closure=optimizer_closure)
        self.lr_schedulers().step()
        self.log("lr", self.lr_schedulers().get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def on_epoch_start(self):
        self.stage1.eval()