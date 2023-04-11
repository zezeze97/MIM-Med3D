import torch
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from torch.nn import BCEWithLogitsLoss
from monai.data import decollate_batch
from monai.transforms import Compose, Activations, AsDiscrete, EnsureType
import sys
sys.path.insert(0,'./code')
from models import UNETR, SwinUNETR
import optimizers
import data


class SegmentationNetTrainer(pl.LightningModule):
    """Pretraining on 3D Imaging with Origin Segmentation Network with ssl task"""

    def __init__(
        self, model_name: str, model_dict: dict,
    ):
        super().__init__()
        self.model_name = model_name
        self.model_dict = model_dict
        if self.model_name == 'unetr':
            self.model = UNETR(**model_dict)
        elif self.model_name == 'swin_unetr':
            self.model = SwinUNETR(**model_dict)

        self.partition_loss = BCEWithLogitsLoss()
        self.post_trans = Compose(
            [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
        )

    def training_step(self, batch, batch_idx):
        # --------------------------
        image_ori = batch["image"]
        batch_size = image_ori.shape[0]
        # mix up input sample
        image_a = image_ori.clone().detach()
        image_b = image_ori.clone().detach()
        random_idx = torch.randperm(batch_size, device=image_ori.device, requires_grad=False)
        image_b = image_b[random_idx, :, :, :, :]
        mix_image = image_a + image_b
        # process overlap
        mix_image[mix_image==2.] = 1.
        mix_image = mix_image.clone().detach().requires_grad_(False)
        mix_image = mix_image.float()
        # get_target
        target = torch.concat([image_a, image_b], dim=1).clone().detach().requires_grad_(False)
        
        # model forward pass
        pred_logits = self.model(mix_image)
        loss = self.partition_loss(pred_logits, target)

        self.log("train/partition_loss", loss, 
                batch_size=batch_size,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # --------------------------
        image_ori = batch["image"]
        batch_size = image_ori.shape[0]
        # mix up input sample
        image_a = image_ori.clone().detach()
        image_b = image_ori.clone().detach()
        random_idx = torch.randperm(batch_size, device=image_ori.device, requires_grad=False)
        image_b = image_b[random_idx, :, :, :, :]
        mix_image = image_a + image_b
        # process overlap
        mix_image[mix_image==2.] = 1.
        mix_image = mix_image.clone().detach().requires_grad_(False)
        mix_image = mix_image.float()
        # get_target (B, 2, x, x, x)
        target = torch.concat([image_a, image_b], dim=1).clone().detach().requires_grad_(False)
        
        # model forward pass
        pred_logits = self.model(mix_image)
        loss = self.partition_loss(pred_logits, target)
        pred_val = self.post_trans(pred_logits)
        acc = torch.mean((pred_val==target).float())

        self.log("val/partition_loss", loss, 
                batch_size=batch_size,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                sync_dist=True)
        self.log("val/acc", acc, 
                batch_size=batch_size,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                sync_dist=True)

        return {"val_loss": loss, "val_acc": acc, "val_number": batch_size}

    def on_validation_epoch_end(self, outputs):
        val_loss, val_acc, num_items = 0, 0, 0
        for output in outputs:
            val_loss += output["val_loss"]
            val_acc += output["val_acc"]
            num_items += output["val_number"]
        mean_val_loss = val_loss / len(outputs)
        mean_val_acc = val_acc / len(outputs)
        self.log(
            "val/partition_loss_avg", mean_val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(
            "val/partition_acc_avg", mean_val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.logger.log_hyperparams(
            params={
                "model": self.model_name,
                **self.model_dict,
                "batch_size": self.trainer.datamodule.batch_size,
                "distribution": self.trainer.datamodule.dist,
                "max_epochs": self.trainer.max_epochs,
                "precision": self.trainer.precision,
            },
            metrics={"partition_loss": mean_val_loss,
                     'partition_acc': mean_val_acc},
        )


if __name__ == "__main__":
    cli = LightningCLI(save_config_kwargs={'overwrite':True})
