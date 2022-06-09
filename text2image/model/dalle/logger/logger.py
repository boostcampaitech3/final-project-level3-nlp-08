import os
import torch
import torchvision
from datetime import datetime
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import rank_zero_only



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


def setup_callbacks(config,result_path,config_downstream):
    # Setup callbacks
    now = datetime.now().strftime('%d%m%Y_%H%M%S')
    result_path = os.path.join(result_path,
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
    logger_img = ImageLogger()
    return checkpoint_callback, logger, logger_img