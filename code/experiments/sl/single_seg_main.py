from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.data import decollate_batch

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
import sys
sys.path.insert(0,'./code')
from models import UNETR, SwinUNETR
import optimizers
import data



class SingleSegtrainer(pl.LightningModule):
    def __init__(self, num_classes: int, model_name: str, model_dict: dict):
        super().__init__()
        self.model_name = model_name
        self.model_dict = model_dict

        if model_name.split("_")[0] == "unetr":
            self.model = UNETR(**model_dict)
        elif model_name == "segresnet":
            self.model = SegResNet(**model_dict)
        elif model_name == "swin_unetr":
            self.model = SwinUNETR(**model_dict)

        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
        self.post_label = AsDiscrete(to_onehot=num_classes)
        self.dice_metric = DiceMetric(
            include_background=True, reduction="mean", get_not_nans=False
        )
        # self.dice_vals = []
        self.metric_values = []
        self.epoch_loss_values = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        batch_size = images.shape[0]
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        # logging
        self.log(
            "train/dice_loss_step",
            loss,
            batch_size=batch_size,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(
            "train/dice_loss_avg",
            avg_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        self.epoch_loss_values.append(avg_loss.detach().cpu().numpy())

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        batch_size = images.shape[0]
        roi_size = (96, 96, 96)
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images,
            roi_size,
            sw_batch_size,
            self.forward,  # the output image will be cropped to the original image size
        )
        loss = self.loss_function(outputs, labels)
        # compute dice score
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        dice = self.dice_metric.aggregate().item()
        # self.dice_metric.reset()
        # compute mean dice score per validation epoch
        # self.dice_vals.append(dice)
        # logging
        self.log(
            "val/dice_loss_step",
            loss,
            batch_size=batch_size,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )

        return {"val_loss": loss, "val_number": len(outputs), "dice": dice}

    def on_validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        dice_vals = []
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
            dice_vals.append(output["dice"])
        mean_val_dice = np.mean(dice_vals)
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        # logging
        self.log(
            "val/dice_loss_avg",
            mean_val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        self.log(
            "val/dice_score_avg",
            mean_val_dice,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        self.logger.log_hyperparams(
            params={
                "model": self.model_name,
                **self.model_dict,
                "batch_size": self.trainer.datamodule.batch_size,
                "distribution": self.trainer.datamodule.dist,
                "max_epochs": self.trainer.max_epochs,
                "precision": self.trainer.precision,
            },
            metrics={"dice_loss": mean_val_loss, "dice_score": mean_val_dice},
        )

    
        self.metric_values.append(mean_val_dice)

    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        batch_size = images.shape[0]
        roi_size = (96, 96, 96)
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images,
            roi_size,
            sw_batch_size,
            self.forward,  # the output image will be cropped to the original image size
        )
        loss = self.loss_function(outputs, labels)
        # compute dice score
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        dice = self.dice_metric.aggregate().item()

        return {"dice": dice}

    def test_epoch_end(self, outputs):
        dice_vals = []
        for output in outputs:
            dice_vals.append(output["dice"])
        mean_val_dice = np.mean(dice_vals)
        # mean_val_dice = self.dice_metric_test.aggregate().item()
        self.dice_metric.reset()

        print(f"avg dice score: {mean_val_dice} ")
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch["image"]
        batch_size = images.shape[0]
        roi_size = (96, 96, 96)
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images,
            roi_size,
            sw_batch_size,
            self.forward,  # the output image will be cropped to the original image size
        )
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        return outputs

if __name__ == "__main__":
    cli = LightningCLI(save_config_kwargs={'overwrite':True})
