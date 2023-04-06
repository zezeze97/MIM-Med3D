import logging
from typing import Union, Sequence

import pytorch_lightning as pl
import os
import json
import torch.distributed as dist

from monai.data import (
    CacheDataset,
    Dataset,
    partition_dataset,
    DataLoader,
    PersistentDataset,
    load_decathlon_datalist,
)
from monai.transforms import (
    Compose,
    AddChanneld,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandScaleIntensityd,
    Spacingd,
    RandShiftIntensityd,
    CropForegroundd,
    SpatialPadd,
    RandSpatialCropSamplesd,
    RandCropByPosNegLabeld,
    MapTransform,
    ConvertToMultiChannelBasedOnBratsClassesd,
    RandSpatialCropd,
    RandFlipd,
    ToTensord
)
from monai.data.utils import pad_list_data_collate

# from .utils import ConvertToMultiChannelBasedOnBratsClassesd, StackStuff, get_modalities



def datafold_read(datalist, basedir, fold=0, key="training"):

    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


# @DATAMODULE_REGISTRY
class BratsDataset(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        json_path: str,
        cache_dir: str,
        fold: int,
        batch_size: int = 1,
        val_batch_size: int = 1,
        num_workers: int = 8,
        cache_num: int = 0,
        cache_rate: float = 0.0,
        spatial_size: Sequence[int] = (96, 96, 96),
        dist: bool = False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.json_path = json_path
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.cache_num = cache_num
        self.cache_rate = cache_rate
        self.spatial_size = spatial_size
        self.dist = dist

        self.train_list, self.valid_list = datafold_read(self.json_path, root_dir, fold)


    def train_transforms(self):
        transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            CropForegroundd(
                keys=["image", "label"], source_key="image", k_divisible=[self.spatial_size[0], self.spatial_size[1], self.spatial_size[2]]
            ),
            RandSpatialCropd(
                keys=["image", "label"], roi_size=[self.spatial_size[0], self.spatial_size[1], self.spatial_size[2]], random_size=False
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ToTensord(keys=["image", "label"]),
        ]
    )
        return transforms

    def val_transforms(self):
        transforms = Compose(
                            [LoadImaged(keys=["image", "label"]),
                            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                            ToTensord(keys=["image", "label"]),
                            ]
                        )
        return transforms

    def test_transforms(self):
        transforms = Compose(
                            [LoadImaged(keys=["image", "label"]),
                            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                            ToTensord(keys=["image", "label"]),
                            ]
                        )
        return transforms

    def setup(self, stage=None):
        if stage in [None, "fit"]:
            if self.dist:
                train_partition = partition_dataset(
                    data=self.train_list,
                    num_partitions=dist.get_world_size(),
                    shuffle=True,
                    even_divisible=True,
                )[dist.get_rank()]

                valid_partition = partition_dataset(
                    data=self.valid_list,
                    num_partitions=dist.get_world_size(),
                    shuffle=False,
                    even_divisible=True,
                )[dist.get_rank()]
            else:
                train_partition = self.train_list
                valid_partition = self.valid_list

            if any([self.cache_num, self.cache_rate]) > 0:
                self.train_ds = CacheDataset(
                    train_partition,
                    cache_num=self.cache_num,
                    cache_rate=self.cache_rate,
                    num_workers=self.num_workers,
                    transform=self.train_transforms(),
                )
                self.valid_ds = CacheDataset(
                    valid_partition,
                    cache_num=self.cache_num,
                    cache_rate=self.cache_rate,
                    num_workers=self.num_workers,
                    transform=self.val_transforms(),
                )
            else:
                logging.info("Loading Persistent Dataset...")
                self.train_ds = PersistentDataset(
                    train_partition,
                    transform=self.train_transforms(),
                    cache_dir=self.cache_dir,
                )
                self.valid_ds = PersistentDataset(
                    valid_partition,
                    transform=self.val_transforms(),
                    cache_dir=self.cache_dir,
                )
        elif stage in [None, 'test']:
            test_partition = self.valid_list
            self.test_ds = PersistentDataset(test_partition,
                                            transform=self.test_transforms(),
                                            cache_dir=self.cache_dir,)
        # self.test_ds = Dataset(self.test_dict, transform=self.test_transforms())

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            collate_fn=pad_list_data_collate,
            # prefetch_factor=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=pad_list_data_collate,
            shuffle=False,
            drop_last=False,
            # prefetch_factor=4,
        )

    def test_dataloader(self):
         return DataLoader(
             self.test_ds, 
             batch_size=1, 
             num_workers=self.num_workers, 
             pin_memory=True,
             shuffle=False,
             drop_last=False,
         )
