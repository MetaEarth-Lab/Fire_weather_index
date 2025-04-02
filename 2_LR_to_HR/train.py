from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.CLIFF_model import *
from datasets.dataset import *
from utils.main_utils import *
from utils.loss_utils import *
from utils.parser import create_parser

import numpy as np
import xarray as xr

import gc
import ast
import warnings
warnings.filterwarnings("ignore")

def load_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path)
    start_epoch = ckpt['epoch']
    model_state_dict = ckpt['model_state_dict']
    optimizer_state_dict = ckpt['optimizer_state_dict']
    scheduler_state_dict = ckpt['scheduler_state_dict']

    # print(f"{ '-' * 30} load ckpt : {ckpt_path} {'-' * 30}")
    print_log(f"{ '-' * 30} load ckpt : {ckpt_path} {'-' * 30}")
    return start_epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict


def train(train_dataset, val_dataset, args, GCM):
    train_batch = args['batch_size']
    val_batch = args['val_batch_size']
    epochs = args['epochs']
    scenario = args['scenario']
    res_dir = args['res_dir']
    res_dir = os.path.join(res_dir, scenario, GCM)
    os.makedirs(res_dir, exist_ok=True)

    start_epoch = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIFF().to(device)
    dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch)
    val_loader = DataLoader(dataset=val_dataset, batch_size=val_batch)
    criterion = NudgingLoss(hr_weight=0.5, lr_weight=0.5)
    optimizer = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    scaler = GradScaler()

    start_logging(prefix="train", args=args)
    if args['ckpt_path'] is not None:
        start_epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict = load_ckpt(args['ckpt_path'])
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        scheduler.load_state_dict(scheduler_state_dict)

    mean = train_dataset.mean.to(device).unsqueeze(0)
    std = train_dataset.std.to(device).unsqueeze(0)

    zmean = train_dataset.zmean.to(device).unsqueeze(0)
    zstd = train_dataset.zstd.to(device).unsqueeze(0)

    lat = np.load("../data/ISIMIP_ko_lat.npy")
    coslat = np.cos(np.deg2rad(np.tile(lat, 25).reshape(25, -1).T))
    coslat = torch.tensor(coslat, device=device)
    coslat = coslat.view(1, 1, 25, 25)
    print_log(f"{ '-' * 30} Scenario : {scenario} / GCM : {GCM} { '-' * 30}")

    for epoch in range(start_epoch, epochs):
        progress_bar = tqdm(dataloader,
                            desc=f"Epoch {epoch+1}/{epochs}",)
        model.train()
        tot_loss = 0
        for batch in progress_bar:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            topo_lr = batch["topo_lr"].to(device)
            topo_hr = batch["topo_hr"].to(device)
            lmask_lr = batch["lmask_lr"].to(device)
            lmask_hr = batch["lmask_hr"].to(device)
            toy = batch["toy"].to(device)

            cell = batch["cell"].to(device)
            hr_coord = batch["coord"].to(device)
            optimizer.zero_grad()
            out = model(lr, hr_coord, cell, topo_lr, topo_hr.view(-1, 25 * 25, 1), lmask_lr, lmask_hr.view(-1, 25 * 25, 1), toy, mean, std, zstd)
            lr_target = model.pool(hr)
            loss, main, hr_loss, lr_loss = criterion(out, hr, out - model.pool(out), hr - lr_target, model.pool(out), lr_target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tot_loss += loss.item()
            progress_bar.set_postfix(
                loss=f"total: {1e4 * loss.item():.3f}, main: {1e4 * main.item():.3f}, hr: {1e4 * hr_loss.item():.3f}, lr: {1e4 * lr_loss.item():.5f}"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        # print(f"Average Training Loss : {tot_loss / len(dataloader):.6f}")
        print_log(f"Average Training Loss : {tot_loss / len(dataloader):.6f}")
        if scheduler:
            scheduler.step()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        }

        torch.save(checkpoint, os.path.join(res_dir, f"liif_{epoch+1}.pt"))
        print_log(f"Save checkpoint {os.path.join(res_dir, f"liif_{epoch+1}.pt")}")
    #     wandb.log({"train_loss": tot_loss / len(dataloader)}, step=epoch)

        model.eval()
        val_running_loss = 0.0
        progress_bar2 = tqdm(val_loader,
                            desc=f"Epoch {epoch+1}/{epochs}",)
        with torch.no_grad():
            for batch in progress_bar2:
                lr = batch["lr"].to(device)
                hr = batch["hr"].to(device)
                topo_lr = batch["topo_lr"].to(device)
                topo_hr = batch["topo_hr"].to(device)
                lmask_lr = batch["lmask_lr"].to(device)
                lmask_hr = batch["lmask_hr"].to(device)
                toy = batch["toy"].to(device)

                cell = batch["cell"].to(device)
                hr_coord = batch["coord"].to(device)

                out = model(lr, hr_coord, cell, topo_lr, topo_hr.view(-1, 25 * 25, 1), lmask_lr, lmask_hr.view(-1, 25 * 25, 1), toy, mean, std, zstd)
                lr_target = model.pool(hr)
                # print("out.shape, hr.shape, lr_target.shape", out.shape, hr.shape, lr_target.shape)
                loss, main, hr_loss, lr_loss = criterion(out, hr, out - model.pool(out), hr - lr_target, model.pool(out), lr_target) #pred, target, hr_pred, hr_target, lr_pred, lr_target

                progress_bar2.set_postfix(
                    loss=f"total: {1e4 * loss.item():.3f}, main: {1e4 * main.item():.3f}, hr: {1e4 * hr_loss.item():.3f}, lr: {1e4 * lr_loss.item():.06f}"
                )
                val_running_loss += loss.item()

        avg_val_loss = val_running_loss / len(val_loader)
        # print(f"Average Validation Loss : {avg_val_loss:.6f}")
        print_log(f"Average Validation Loss : {avg_val_loss:.6f}")

        torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':

    args = create_parser().parse_args()
    args = args.__dict__
    config_file = args['config_file']
    config = load_config(config_file)
    args = update_config(config, args)

    val_years = args['val_years']
    train_years = args['train_years']
    GCMs_list = args['gcms']
    base_path = args['base_path']
    mean_std_path = args['mean_std_path']
    topo_path = args['topo_path']
    lmask_path = args['lmask_path']

    for idx in range(len(GCMs_list)):
        GCM = GCMs_list[idx]

        train_dataset = SRDataset(
            input_dir=f"{base_path}/{GCM}",
            years=train_years,
            mean_std_path = mean_std_path,
            topo_path=topo_path,
            lmask_path = lmask_path
        )
        val_dataset = SRDataset(
            input_dir=f"{base_path}/{GCM}",
            years=val_years,
            mean_std_path = mean_std_path,
            topo_path=topo_path,
            lmask_path = lmask_path
        )

        train(train_dataset, val_dataset, args, GCM)
    
