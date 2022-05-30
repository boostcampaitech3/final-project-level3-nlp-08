import os
import numpy as np
import pandas as pd
from PIL import Image

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils
import random
from torchvision.transforms import Resize, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler
import sys
import argparse
from typing import Optional
from datetime import datetime
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.distributed import rank_zero_only

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dalle.models import ImageGPT, Rep_Dalle

seed = 42
path_upstream = 'minDALL-E/1.3B'
exp_name = 'exp2_ep4'
config_file = './configs/'+exp_name+'.yaml'
config_downstream = config_file
result_path = './'+exp_name
data_dir = './img_data'
n_gpus = 1

train, val = None, None

def seed_everything(seed):
    """
    동일한 조건으로 학습을 할 때, 동일한 결과를 얻기 위해 seed를 고정시킵니다.

    Args:
        seed: seed 정수값
    """
    torch.manual_seed(seed)  # pytorch의 random seed 고정
    torch.cuda.manual_seed(seed)  # GPU 에서 사용하는 난수 생성 시드 고정
    # torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True  # CuDNN 부분고정
    torch.backends.cudnn.benchmark = False  # CuDNN 부분고정
    np.random.seed(seed)  # Numpy 부분
    random.seed(seed)  # transforms에서 random 라이브러리를 사용하기 때문에 random 라이브러리를 불러서 고정
def search(dirname, result):  # 하위목록의 모든 파일을 찾는 함수
    try:
        filenames = os.listdir(dirname)
        print(f'file 개수 : {len(filenames)}')
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                if full_filename.startswith('.'):
                    continue
                search(full_filename, result)
            else:
                ext = os.path.splitext(full_filename)[-1]  # 확장자 체크
                if ext:
                    result.append(full_filename)
                else:
                    print(full_filename)
    except PermissionError:
        print('error')
def labeling(dirname, result, prefix):  # 라벨링하는 함수
    try:
        filenames = os.listdir(dirname)
        for filename in filenames:
            if filename.startswith('.'):
                continue
            tmp_str = filename.split(".")[0].split("__")[0]
            keyword = prefix + ' '.join(tmp_str.split("-")[:-1])
            result.append(keyword)
    except PermissionError:
        print('error')
def check_unopen_img(df):
    for index in range(len(df)):
        try:
            image = Image.open(df['path'].iloc[index]).convert('RGB')
            label = df['label'].iloc[index]
        except e:
            print(e)
            print(f'Image load error : {df["path"].iloc[index]}')
def rm_unopen_img(df):
    for index in range(len(df)):
        try:
            image = Image.open(df['path'].iloc[index]).convert('RGB')
            label = df['label'].iloc[index]
        except:
            print(f'Image load error : {df["path"].iloc[index]}')
            os.remove(df["path"].iloc[index])

class CustomDataset(Dataset):
    def __init__(self, img_paths_label, transform):
        self.X = img_paths_label['path']
        self.y = img_paths_label['label']
        self.transform = transform

    def __getitem__(self, index):
        # image = Image.open(self.X.iloc[index])
        try:
            image = Image.open(self.X.iloc[index]).convert('RGB')
            label = self.y.iloc[index]
            if self.transform:
                image = self.transform(image)
            return image, label
        except:
            print(f'Image load error : {self.X.iloc[index]}')

    def __len__(self):
        return len(self.X)
class ImageLogger(Callback):
    def __init__(self):
        super().__init__()

    @rank_zero_only
    def log_img(self, pl_module, batch, current_epoch, split="train"):
        with torch.no_grad():
            images, labels = batch
            recons = pl_module.stage1(images)
            images = images.cpu()
            recons = recons.cpu()

            grid_org = (torchvision.utils.make_grid(images, nrow=8) + 1.0) / 2.0
            grid_rec = (torchvision.utils.make_grid(recons, nrow=8) + 1.0) / 2.0
            grid_rec = torch.clip(grid_rec, min=0, max=1)

            pl_module.logger.experiment.add_image(f"images_org/{split}", grid_org, global_step=current_epoch)
            pl_module.logger.experiment.add_image(f"images_rec/{split}", grid_rec, global_step=current_epoch)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0 and trainer.current_epoch < 5:
            self.log_img(pl_module, batch, current_epoch=trainer.current_epoch, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0 and trainer.current_epoch < 5:
            self.log_img(pl_module, batch, current_epoch=trainer.current_epoch, split="test")
class CustomDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: Optional[str] = None,
                 image_resolution: int = 256,
                 train_batch_size: int = 2,
                 valid_batch_size: int = 32,
                 num_workers: int = 8):
        super().__init__()

        self.data_dir = data_dir
        self.image_resolution = image_resolution
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.num_workers = num_workers

        self.train_transform = transforms.Compose(
            [transforms.Resize(image_resolution),
             transforms.RandomCrop(image_resolution),
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )
        self.valid_transform = transforms.Compose(
            [transforms.Resize(image_resolution),
             transforms.CenterCrop(image_resolution),
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )

    def setup(self, stage=None):
        self.trainset = CustomDataset(train, data_transforms['train'])
        self.validset = CustomDataset(valid, data_transforms['val'])

    def train_dataloader(self):
        return DataLoader(self.trainset,
                          batch_size=self.train_batch_size,
                          # num_workers=0,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def valid_dataloader(self):
        return DataLoader(self.validset,
                          batch_size=self.valid_batch_size,
                          # num_workers=0,
                          num_workers=self.num_workers,
                          pin_memory=True)
def setup_callbacks(config, args_result_path):
    # Setup callbacks
    now = datetime.now().strftime('%d%m%Y_%H%M%S')
    result_path = os.path.join(args_result_path,
                               os.path.basename(config_downstream).split('.')[0],
                               now)
    ckpt_path = os.path.join(result_path, 'ckpt')
    log_path = os.path.join(result_path, 'log')

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        filename="custom-clscond-gen-{epoch:02d}" if config.stage2.use_cls_cond else
                 "custom-uncond-gen-{epoch:02d}",
        every_n_epochs=config.experiment.save_ckpt_freq,
        save_weights_only=True,
        save_last=True
    )
    logger = TensorBoardLogger(log_path, name="iGPT")
    # logger = WandbLogger(name='minDALL-E_ep3',project='final',save_dir=log_path, name="iGPT")
    logger_img = ImageLogger()
    return checkpoint_callback, logger, logger_img

def mk_dataframe():
    seed_everything(seed)
    illust_all_path = []
    search("./img_data/illustrations", illust_all_path)
    scenery_all_path = []
    search("./img_data/scenery", scenery_all_path)
    vector_all_path = []
    search("./img_data/vectors", vector_all_path)
    dirname, illust_label = "./img_data/illustrations", []
    labeling(dirname, illust_label, "an illustration image of ")
    dirname, vector_label = "./img_data/vectors", []
    labeling(dirname, vector_label, "a vector image of ")
    dirname, scenery_label = "./img_data/scenery", []
    labeling(dirname, scenery_label, "a scenery of ")

    illust_df = pd.DataFrame(illust_all_path, columns=['path'])
    illust_df['label'] = illust_label

    scenery_df = pd.DataFrame(scenery_all_path, columns=['path'])
    scenery_df['label'] = scenery_label

    vector_df = pd.DataFrame(vector_all_path, columns=['path'])
    vector_df['label'] = vector_label

    df = pd.concat([illust_df, vector_df, scenery_df], ignore_index=True)

    # if you want to check or remove wrong images, run below method. Then rerun mk_dataframe.
    # check_unopen_img(df)
    # rm_unopen_img(df)
    return df

def main():
    torch.cuda.empty_cache()
    # mk dataframe
    df = mk_dataframe()

    # mk Custom dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    global data_transforms
    data_transforms = {
        'train':
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                normalize
            ]),
        'val':
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                normalize
            ]),
    }

    global train, valid
    train, valid = train_test_split(df, test_size=0.2,
                                    shuffle=True,
                                    random_state=42)

    pl.seed_everything(seed)

    # Build iGPT
    model, config = Rep_Dalle.from_pretrained(path_upstream, config_downstream)

    # Setup callbacks
    ckpt_callback, logger, logger_img = setup_callbacks(config, result_path)

    # Build data modules
    dataset = CustomDataModule(data_dir=data_dir,
                               image_resolution=config.dataset.image_resolution,
                               train_batch_size=config.experiment.local_batch_size,
                               valid_batch_size=config.experiment.valid_batch_size,
                               num_workers=16)
    dataset.setup()
    train_dataloader = dataset.train_dataloader()
    valid_dataloader = dataset.valid_dataloader()
    print(f"len(train_dataset) = {len(dataset.trainset)}")
    print(f"len(valid_dataset) = {len(dataset.validset)}")

    # Calculate how many batches are accumulated
    assert config.experiment.total_batch_size % (config.experiment.local_batch_size * n_gpus) == 0
    grad_accm_steps = config.experiment.total_batch_size // (config.experiment.local_batch_size * n_gpus)
    config.optimizer.max_steps = len(dataset.trainset) // config.experiment.total_batch_size * config.experiment.epochs

    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        min_delta=0.00,
        patience=3,
        verbose=True,
        mode='min'
    )
    # Build trainer
    trainer = pl.Trainer(max_epochs=config.experiment.epochs,
                         accumulate_grad_batches=grad_accm_steps,
                         gradient_clip_val=config.optimizer.grad_clip_norm,
                         precision=16 if config.experiment.use_amp else 32,
                         callbacks=[ckpt_callback, logger_img,early_stop_callback],
                         accelerator="gpu",
                         devices=n_gpus,
                         # strategy="ddp",
                         logger=logger,
                         )
    trainer.fit(model, train_dataloader, valid_dataloader)

if __name__ == '__main__':
    main()






