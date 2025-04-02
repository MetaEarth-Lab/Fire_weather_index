from utils.main_utils import make_coord

import os
import numpy as np
import xarray as xr

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pickle

class SRDataset(Dataset):
    def __init__(self, input_dir, years, mean_std_path, topo_path, lmask_path):
        self.input_dir = input_dir
        self.vals = ['tasmax', 'huss', 'ps', 'pr', 'rlds', 'rsds', 'sfcwind']
        self.GCM = input_dir.split("/")[-1]
        self.samples = []

        self.topo = xr.open_dataarray(topo_path).values[::-1]
        self.topo = torch.from_numpy(self.topo.copy()).float().unsqueeze(0)
        self.lmask = xr.open_dataarray(lmask_path).values[::-1]
        self.lmask = torch.from_numpy(self.lmask.copy()).float().unsqueeze(0)

        # mean = np.load(mean_path)
        # std = np.load(std_path)

        # Load pre-computed global mean and std
        with open(mean_std_path, 'rb') as fr:
            mean_std_dict = pickle.load(fr)

        mean, std = [], []
        for val in self.vals:
            mean.append(mean_std_dict[self.GCM][val]['mean'])
            std.append(mean_std_dict[self.GCM][val]['std'])

        mean = np.stack(mean, axis=0)
        std = np.stack(std, axis=0)

        # Convert to torch and get channel-wise mean/std
        self.mean = torch.from_numpy(mean.mean(axis=(1, 2))).float().view(-1, 1, 1)  # [8, 1, 1]
        self.std = torch.from_numpy(std.mean(axis=(1, 2))).float().view(-1, 1, 1)    # [8, 1, 1]

        self.zmean = self.topo.mean()
        self.zstd = self.topo.std()
        
        for year in years:
            file_path = os.path.join(input_dir, f"{year}.nc")
            if os.path.exists(file_path):
                with xr.open_dataset(file_path, chunks="auto") as ds:
                    for t in range(len(ds.time)):
                        toy = torch.tensor(2 * np.pi * t / len(ds.time)).float()
                        self.samples.append((file_path, t, toy))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, t, toy = self.samples[idx]

        with xr.open_dataset(file_path) as ds:
            ds = ds.chunk({"time":1})
            data = ds.to_array().isel(time=t).values
        
        data[1] = np.log(1 + data[1] * 1000) #huss
        data[2] = np.log(data[2] / 100) #ps
        data[3] = np.log(1 + data[3] * 86400) # pr
        data[4] = np.log(data[4]) # rlds
        data[5] = np.log(1 + data[5]) #rsds

        sample = torch.tensor(data, dtype=torch.float32)

        sample = (sample - self.mean) / self.std
        topo = (self.topo - self.zmean) / self.zstd
        lmask = 2 * (self.lmask - 0.5)

        sample_lr = F.interpolate(sample.unsqueeze(0), scale_factor=0.5, mode="bilinear").squeeze(0)
        topo_lr = F.interpolate(topo.unsqueeze(0), size=(12,12),mode="bilinear").squeeze(0)
        lmask_lr = F.interpolate(lmask.unsqueeze(0), size=(12,12), mode="bilinear").squeeze(0)

        coord = make_coord((601,601))
        cell = torch.ones_like(coord)
        cell[:,0] *= 2 / sample.shape[-2]
        cell[:,1] *= 2 / sample.shape[-1]

        _toy = torch.ones_like(coord)
        _toy[:,0] = torch.sin(toy)
        _toy[:,1] = torch.cos(toy)

        return {
            "hr": sample,
            "lr": sample_lr,
            "topo_hr": topo,
            "topo_lr": topo_lr,
            "lmask_hr": lmask,
            "lmask_lr": lmask_lr,
            "toy": _toy,
            "cell": cell,
            "coord": coord
        }
