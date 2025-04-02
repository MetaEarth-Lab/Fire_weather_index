import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.main_utils import make_coord

class GeoPad(nn.Module):
    def __init__(self, pad_width: int | tuple):
        super().__init__()
        if isinstance(pad_width, int):
            self.padx = self.pady = pad_width
        elif isinstance(pad_width, tuple):
            self.pady, self.padx = pad_width

    def forward(self, x):
        im = x.shape[-1] // 2
        if self.pady > 0:
            top = torch.cat([x[..., :self.pady, im:], x[..., :self.pady, :im]], dim=-1)
            bot = torch.cat([x[..., -self.pady:, im:], x[..., -self.pady:, :im]], dim=-1)
            x = torch.cat([top.flip(-2), x, bot.flip(-2)], dim=-2)
        out = torch.cat([x[..., -self.padx:], x, x[..., :self.padx]], dim=-1) if self.padx > 0 else x
        return out

## RDN backbone model for feature extraction (encoder)

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3, use_geo=True):
        super().__init__()
        inC = inChannels
        G = growRate
        if use_geo:
            self.conv = nn.Sequential(
                GeoPad((kSize - 1) // 2),
                nn.Conv2d(inC, G, kSize),
                nn.ReLU()
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inC, G, kSize, 1, (kSize - 1) // 2),
                nn.ReLU()
            )

    def forward(self, x):
        out = self.conv(x)
        return torch.cat([x, out], dim=1)
    
class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3, use_geo=True):
        super().__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G, kSize, use_geo=use_geo))
        self.convs = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x
    
class RDN(nn.Module):
    def __init__(self, nf, G0, kSize, use_geo=True):
        super().__init__()
        self.D, C, G = (16, 8, 64)

        if use_geo:
            self.SFENet1 = nn.Sequential(
                GeoPad((kSize - 1) // 2),
                nn.Conv2d(nf + 1, G0, kSize)
            )
            self.SFENet2 = nn.Sequential(
                GeoPad((kSize - 1) // 2),
                nn.Conv2d(G0, G0, kSize)
            )
            self.GFF = nn.Sequential(
                nn.Conv2d(self.D * G0, G0, 1),
                GeoPad((kSize - 1) // 2),
                nn.Conv2d(G0, G0, kSize)
            )
        else:
            self.SFENet1 = nn.Conv2d(nf + 1, G0, kSize, 1, (kSize - 1) // 2)
            self.SFENet2 = nn.Conv2d(G0, G0, kSize, 1, (kSize - 1) // 2)
            self.GFF = nn.Sequential(
                nn.Conv2d(self.D * G0, G0, 1),
                nn.Conv2d(G0, G0, kSize, 1, (kSize - 1) // 2)
            )

        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C, use_geo=use_geo)
            )

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1
        return x

## LIFF based topographic downscaling model (encoder + decoder)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]

        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

class CLIFF(nn.Module):
    def __init__(self, nf=7, use_geo=False):
        super().__init__()
        self.use_geo = use_geo
        self.encoder = RDN(nf=nf, G0=64, kSize=3, use_geo=use_geo)
        self.imnet = MLP(64 * 9 + 2 + 2 + 8, nf, [256, 256, 256, 256])
        if use_geo:
            self.pool = nn.Sequential(
                GeoPad(1),
                nn.AvgPool2d(kernel_size=3, stride=1)
            )
        else:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def get_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query(self, coord, cell, topo_lr, topo_hr, lmask_lr, lmask_hr, toy):
        feat = self.feat
        if self.use_geo:
            feat = F.unfold(GeoPad(1)(feat), 3).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
        else:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps = 1e-6

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = (
            make_coord(feat.shape[-2:], flatten=False).cuda()
            .permute(2, 0, 1).unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        )

        preds = []
        areas = []

        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps
                coord_[:, :, 1] += vy * ry + eps
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode="nearest", align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_topo = F.grid_sample(
                    topo_lr, coord_.flip(-1).unsqueeze(1),
                    mode="nearest", align_corners=False)[..., 0, :] \
                    .permute(0, 2, 1)
                q_lmask = F.grid_sample(
                    lmask_lr, coord_.flip(-1).unsqueeze(1),
                    mode="nearest", align_corners=False)[..., 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode="nearest", align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                
                rel_coord = coord - q_coord
                rel_topo = topo_hr - q_topo
                # print(lmask_hr.shape, q_lmask.shape)
                rel_lmask = lmask_hr - q_lmask

                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                inp = torch.cat([q_feat, rel_coord, 
                                 q_topo, topo_hr, rel_topo,
                                 q_lmask, lmask_hr, rel_lmask, toy], dim=-1)

                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([inp, rel_cell], dim=-1)
                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, inp, coord, cell, 
                topo_lr, topo_hr, lmask_lr, lmask_hr, toy, mean, std, zstd):
        f = inp.shape[1]
        
        self.get_feat(torch.cat([inp, topo_lr], dim=1))

        H = 601
        W = H

        ## topographic correction for residual connection

        xb = F.interpolate(inp, (H, W), mode="bilinear", align_corners=False)
        xb = xb * std + mean
        zb = F.interpolate(topo_lr, (H, W), mode="bilinear", align_corners=False)
        dz = (topo_hr.view(-1, 1, H, W) - zb) * zstd

        ## (index) Tair : 0, Qair : 1, PSurf : 2, LWdown : 4

        T = xb[:,0:1] - dz * 6.5 / 1000
        P = xb[:,2:3] - dz / 287.05 / (T + xb[:,0:1]) * 2
        q = torch.exp(xb[:,1:2]) - 1
        Q = q * torch.exp(17.67 * (T - xb[:,0:1]) / (T - 273.15 + 243.5))
        Q = torch.log(1 + Q)

        e1 = 6.112 * torch.exp(17.67 * (xb[:,0:1] - 273.15) / (T - 273.15 + 243.5))
        e2 = 6.112 * torch.exp(17.67 * (T - 273.15) / (T - 273.15 + 243.5))

        eps1 = 1.08 * (1 - torch.exp(- e1 ** (xb[:,0:1] / 2016)))
        eps2 = 1.08 * (1 - torch.exp(- e2 ** (T / 2016)))

        L = xb[:,4:5] + 4 * torch.log(T / xb[:,0:1]) + torch.log(eps2 / eps1)

        xc = torch.cat([T, Q, P, xb[:,3:4], L, xb[:,5:]], dim=1)
        xc = (xc - mean) / std
        return xc + self.query(coord, cell, topo_lr, topo_hr, lmask_lr, lmask_hr, toy).view(-1, H, W, f).permute(0,3,1,2)