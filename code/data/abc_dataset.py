from typing import Optional, Sequence, Union
import os
import torch
from torch.utils.data import Dataset
import torch.distributed as ptdist
import pytorch_lightning as pl
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from monai.transforms import (
    Compose,
    RandSpatialCrop,
    SpatialPad,
    RandFlip,
    RandRotate90
)


class ABC(Dataset):
    def __init__(self, 
                 root_dir: str, 
                 split: str,
                 convert_size: tuple = (96, 96, 96),
                 ):
        self.root_dir = root_dir
        self.split = split
        self.convert_size = convert_size
        with open(self.split, 'r') as f:
            data_lst = f.readlines()
        self.data_lst = [item.strip('\n') for item in data_lst]
        self.transform = Compose([RandSpatialCrop(roi_size=convert_size,
                                                  random_size=False),
                                    SpatialPad(spatial_size=convert_size,
                                             method="symmetric"),
                                    RandFlip(spatial_axis=[0], prob=0.10,),
                                    RandFlip(spatial_axis=[1], prob=0.10,),
                                    RandFlip(spatial_axis=[2], prob=0.10,),
                                    RandRotate90(prob=0.10, max_k=3,)])
    
    def __len__(self):
        return len(self.data_lst)
    
    def __getitem__(self, index):
        input_file_path = os.path.join(self.root_dir, self.data_lst[index])
        voxel = np.load(input_file_path)
        # print(f'origin shape is {voxel.shape}')
        # convert to torch
        voxel = torch.from_numpy(voxel).float()
        # convert to channel first
        voxel = voxel.unsqueeze(0)
        voxel = self.transform(voxel)
        voxel = voxel.as_tensor()
        # print(f'convert shape is {voxel.shape}')
        return {'image': voxel}
        
        


class ABCDataset(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        convert_size: tuple = (96, 96, 96),
        batch_size: int = 1,
        val_batch_size: int = 1,
        num_workers: int = 4,
        dist: bool = False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.convert_size = convert_size
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.dist = dist



    def setup(self, stage: Optional[str] = None):
        # Assign Train split(s) for use in Dataloaders
        if stage in [None, "fit"]:
            self.train_ds = ABC(root_dir=self.root_dir, 
                                split=os.path.join(self.root_dir, 'train.txt'), 
                                convert_size=self.convert_size)
            self.valid_ds = ABC(root_dir=self.root_dir, 
                                split=os.path.join(self.root_dir, 'val.txt'), 
                                convert_size=self.convert_size)
          

        if stage in [None, "test"]:
            self.test_ds = ABC(root_dir=self.root_dir, 
                               split=os.path.join(self.root_dir, 'val.txt'), 
                               convert_size=self.convert_size)

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
        )
        else:
            dataloader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
        )
            
        return dataloader

    def val_dataloader(self):
        if self.dist:
            dataloader = torch.utils.data.DataLoader(
            self.valid_ds,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(self.valid_ds),
            drop_last=False
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                self.valid_ds,
                batch_size=self.val_batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=False,
                drop_last=False,
            )
        return dataloader
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )


if __name__ =="__main__":
    
    dataset = ABCDataset(
    root_dir="/Users/zhangzeren/Downloads/dataset/abc",
    convert_size=(96, 96, 96),
    batch_size=8,
    val_batch_size=1,
    num_workers=0,
    dist=False,
    )
    dataset.setup()
    for i, item in enumerate(dataset.train_dataloader()):
        print(item["image"].shape)
        # break
        