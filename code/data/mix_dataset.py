from typing import Optional
import os
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data.distributed import DistributedSampler
from .abc_dataset import ABC
from .modelNet_dataset import ModelNet
from .ScanObjectNN_dataset import ScanObjectNN
from torch.utils.data import ConcatDataset
import bisect

class MIX(ConcatDataset):
    def __init__(self, datasets) -> None:
        super().__init__(datasets)
    
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx
        
        


class MixDataset(pl.LightningDataModule):
    def __init__(
        self,
        modelnet40_root_dir: str,
        scanObjNN_root_dir: str,
        ABC_root_dir: str,
        convert_size: tuple = (96, 96, 96),
        batch_size: int = 1,
        val_batch_size: int = 1,
        num_workers: int = 4,
        dist: bool = False,
        json_path = None,
        downsample_ratio=None
    ):
        super().__init__()
        self.modelnet40_root_dir = modelnet40_root_dir
        self.scanObjNN_root_dir = scanObjNN_root_dir
        self.ABC_root_dir = ABC_root_dir
        self.convert_size = convert_size
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.dist = dist
        self.json_path = json_path
        self.downsample_ratio = downsample_ratio



    def setup(self, stage: Optional[str] = None):
        # Assign Train split(s) for use in Dataloaders
        if stage in [None, "fit"]:
            self.train_ds = [ModelNet(root_dir=self.root_dir, split='train', convert_size=self.convert_size),
                             ScanObjectNN(root_dir=self.root_dir, split=self.split, convert_size=self.convert_size, is_train=True),
                             ABC(root_dir=self.root_dir, split='train', convert_size=self.convert_size)]
            self.valid_ds = [ModelNet(root_dir=self.root_dir, split='test', convert_size=self.convert_size),
                             ScanObjectNN(root_dir=self.root_dir, split=self.split, convert_size=self.convert_size, is_train=True),
                             ABC(root_dir=self.root_dir, split='train', convert_size=self.convert_size)]
          

        if stage in [None, "test"]:
            self.test_ds = ABC(root_dir=self.root_dir, split='test', convert_size=self.convert_size)

    def train_dataloader(self):
        if self.dist:
            dataloader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(self.train_ds),
            drop_last=False,
            # collate_fn=pad_list_data_collate,
            # prefetch_factor=4,
        )
        else:
            dataloader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            # sampler=DistributedSampler(self.train_ds),
            drop_last=False,
            # collate_fn=pad_list_data_collate,
            # prefetch_factor=4,
        )
            
        return dataloader

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_ds,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            # sampler=DistributedSampler(self.valid_ds),
            drop_last=False,
            # collate_fn=pad_list_data_collate,
            # prefetch_factor=4,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            # sampler=DistributedSampler(self.test_ds),
            drop_last=False,
            # collate_fn=pad_list_data_collate,
            # prefetch_factor=4,
        )
