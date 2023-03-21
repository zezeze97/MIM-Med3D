from typing import Optional, Sequence, Union
import os
import torch
from torch.utils.data import Dataset
import torch.distributed as ptdist
import pytorch_lightning as pl
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from monai.data import MetaTensor
import stltovoxel
from stl import mesh

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
    
    def __len__(self):
        return len(self.data_lst)
    
    def __getitem__(self, index):
        input_file_path = os.path.join(self.root_dir, self.data_lst[index])
        mesh_obj = mesh.Mesh.from_file(input_file_path)
        org_mesh = np.hstack((mesh_obj.v0[:, np.newaxis], mesh_obj.v1[:, np.newaxis], mesh_obj.v2[:, np.newaxis]))
        voxel, scale, shift = stltovoxel.convert_mesh(org_mesh, resolution=(self.convert_size[0]-1), parallel=True)
        voxel = self.transform(voxel, self.convert_size)
        return {'image': voxel}
    
    def transform(self, voxel, convert_size):
        z_size, h, w = voxel.shape 
        # z_size must equal to self.convert_size[0]
        assert z_size == self.convert_size[0]
        new_voxel = np.zeros(convert_size)
        if h <= self.convert_size[1] and w <= self.convert_size[2]:
            new_voxel[:z_size, :h, :w] = voxel
        elif h<= self.convert_size[1] and w > self.convert_size[2]:
            new_voxel[:z_size, :h, :] = voxel[:,:,:self.convert_size[2]]
        elif h > self.convert_size[1] and w <= self.convert_size[2]:
            new_voxel[:z_size, :, :w] = voxel[:, :self.convert_size[1], :]
        else:
            new_voxel[:z_size, :, :] = voxel[:, :self.convert_size[1], :self.convert_size[2]]
        # add channel: (96, 96, 96) -> (1, 96, 96, 96)
        new_voxel = torch.from_numpy(new_voxel)
        new_voxel = new_voxel.unsqueeze(0)
        return MetaTensor(new_voxel)
        
        


class ABCDataset(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        convert_size: tuple = (96, 96, 96),
        batch_size: int = 1,
        val_batch_size: int = 1,
        num_workers: int = 4,
        dist: bool = False,
        json_path = None,
        downsample_ratio=None
    ):
        super().__init__()
        self.root_dir = root_dir
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
            self.train_ds = ABC(root_dir=self.root_dir, split='train', convert_size=self.convert_size)
            self.valid_ds = ABC(root_dir=self.root_dir, split='test', convert_size=self.convert_size)
          

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
