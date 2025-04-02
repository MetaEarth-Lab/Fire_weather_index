from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.CLIFF_model_test import *
from datasets.dataset_test import *

from utils.main_utils import *
from utils.loss_utils import *
from utils.parser import create_parser

import numpy as np
import xarray as xr

import gc
import warnings
warnings.filterwarnings("ignore")

def test(test_dataset, ckpt_path, result_dir, year):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIFF().to(device)
    state_dict = torch.load(ckpt_path)['model_state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    result_dir = os.path.join(result_dir, ckpt_path.split('/')[-1].split(".")[0])
    os.makedirs(result_dir, exist_ok=True)

    mean = test_dataset.mean.to(device).unsqueeze(0)
    std = test_dataset.std.to(device).unsqueeze(0)

    zmean = test_dataset.zmean.to(device).unsqueeze(0)
    zstd = test_dataset.zstd.to(device).unsqueeze(0)

    lat = np.load("../data/ISIMIP_ko_lat.npy")
    coslat = np.cos(np.deg2rad(np.tile(lat, 25).reshape(25, -1).T))
    coslat = torch.tensor(coslat, device=device)
    coslat = coslat.view(1, 1, 25, 25)

    model.eval()
    progress_bar2 = tqdm(test_loader)

    preds = []
    with torch.no_grad():
        for idx, batch in enumerate(progress_bar2):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            topo_lr = batch["topo_lr"].to(device)
            topo_hr = batch["topo_hr"].to(device)
            lmask_lr = batch["lmask_lr"].to(device)
            lmask_hr = batch["lmask_hr"].to(device)
            toy = batch["toy"].to(device)

            cell = batch["cell"].to(device)
            hr_coord = batch["coord"].to(device)

            out = model(lr, hr_coord, cell, topo_lr, topo_hr.view(-1, 601 * 601, 1), lmask_lr, lmask_hr.view(-1, 601 * 601, 1), toy, mean, std, zstd)#.view(-1, 601, 601, 7).permute(0,3,1,2) #view(-1,4,14,15) 추가
            lr_target = model.pool(hr)

            pred = out[0].cpu().numpy()
            # target = hr
            pred = pred * std[0].cpu().numpy() + mean[0].cpu().numpy() # 정규화 풀기

            pred[1] = (np.exp(pred[1]) - 1) / 1000
            pred[2] = np.exp(pred[2]) * 100
            pred[3] = (np.exp(pred[3]) - 1) / 86400
            pred[4] = np.exp(pred[4])
            pred[5] = np.exp(pred[5]) - 1

            # np.save(os.path.join(result_dir, f"pred_{idx}.npy"), pred)
            preds.append(pred)
            # np.save(os.path.join(result_dir, f"target_{idx}.npy"), target.cpu().numpy())

    preds = np.stack(preds)  # (time, 7, 601, 601)
    
    save_path = os.path.join(result_dir, f"preds_{year}.npy")
    np.save(save_path, preds)
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    args = create_parser().parse_args()
    args = args.__dict__
    config_file = args['config_file']
    config = load_config(config_file)
    args = update_config(config, args)

    test_years = args['test_years']
    # GCMs_list = args['gcms']
    ckpt_path = args['ckpt_path']
    GCM = ckpt_path.split("/")[-2]

    base_path = args['base_path']
    mean_std_path = args['mean_std_path']
    topo_path = args['topo_path']
    lmask_path = args['lmask_path']
    res_dir = args['res_dir']
    res_dir = os.path.join(res_dir, args['scenario'], GCM)
    

    for year in test_years: 
        test_dataset = SRDataset(
            input_dir=f"{base_path}/{GCM}",
            years=[year],
            mean_std_path = mean_std_path,
            topo_path=topo_path,
            lmask_path = lmask_path,
        )

        test(test_dataset, ckpt_path, res_dir, year)
