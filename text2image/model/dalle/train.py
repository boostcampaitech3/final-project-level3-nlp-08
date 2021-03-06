# ------------------------------------------------------------------------------------
# Modified from minDALL-E (https://github.com/kakaobrain/minDALL-E)
# ------------------------------------------------------------------------------------

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('train.py'))))
from dalle.models import Rep_Dalle
from data_loader.dataset import CustomDataModule
from data_loader.dataloader import CustomDataLoader
from data_loader.utils import mk_dataframe, data_setting, check_unopen_img, rm_unopen_img
from logger.logger import setup_callbacks

seed = 42
path_upstream = 'minDALL-E/1.3B'
config_file = '../configs/CALL-E.yaml'
config_downstream = config_file
result_path = '../tf_model'
data_dir = './img_data'
n_gpus = 1
train, val = None, None
cleaning = False

def main():
    """
    transfer learning minDALL-E with Pixabay Custom dataset for text2image
    """
    torch.cuda.empty_cache()
    # mk dataframe
    df = mk_dataframe(seed)
    
    if cleaning:
        check_unopen_img(df)
        rm_unopen_img(df)

    # mk Custom dataset setting : transform variable, simply splited train/valid dataset (valid size=0.2, shuffle=True)
    data_transforms, train, valid = data_setting(df)

    pl.seed_everything(seed)

    # Build iGPT
    model, config = Rep_Dalle.from_pretrained(path_upstream, config_downstream)
    
    # Add config for setup callbacks & data modules
    # config.data_transforms = data_transforms
    # config.train = train
    # config.valid = valid
    # config.result_path = result_path
    # config.config_downstream = config_downstream

    # Setup callbacks
    ckpt_callback, logger, logger_img = setup_callbacks(config=config, result_path=result_path,
                                                        config_downstream=config_downstream)

    # Build data modules
    dataset = CustomDataModule(config=config,
                               data_dir=data_dir,
                               data={'train': train, 'valid': valid},
                               data_transforms=data_transforms,
                               image_resolution=config.dataset.image_resolution,
                               train_batch_size=config.experiment.local_batch_size,
                               valid_batch_size=config.experiment.valid_batch_size,
                               num_workers=16)
    dataset.setup()
    train_dataloader = DataLoader(dataset=dataset.trainset,
                                        batch_size=dataset.train_batch_size,
                                        num_workers=dataset.num_workers,
                                        pin_memory=True)
    valid_dataloader = DataLoader(dataset=dataset.validset,
                                        batch_size=dataset.valid_batch_size,
                                        num_workers=dataset.num_workers,
                                        pin_memory=True)
    
    print(f"len(train_dataset) = {len(dataset.trainset)}")
    print(f"len(valid_dataset) = {len(dataset.validset)}")

    # Calculate how many batches are accumulated
    assert config.experiment.total_batch_size % (config.experiment.local_batch_size * n_gpus) == 0
    grad_accm_steps = config.experiment.total_batch_size // (config.experiment.local_batch_size * n_gpus)
    config.optimizer.max_steps = len(dataset.trainset) // config.experiment.total_batch_size * config.experiment.epochs

    # Setting EarlyStopping
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
                         callbacks=[ckpt_callback, logger_img, early_stop_callback],
                         accelerator="gpu",
                         devices=n_gpus,
                         # strategy="ddp",
                         logger=logger)
    
    trainer.fit(model, train_dataloader, valid_dataloader)

if __name__ == '__main__':
    main()