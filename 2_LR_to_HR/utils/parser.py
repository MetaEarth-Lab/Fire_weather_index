# Copyright (c) CAIRI AI Lab. All rights reserved

import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--res_dir', default='./checkpoint', type=str)
    parser.add_argument('--scenario', '-sn', default='historical', type=str)
    parser.add_argument('--gcms', nargs='+', default=['GFDL-ESM4', 'IPSL-CM6A-LR', 'MPI-ESM1-2-HR',
                                            'MRI-ESM2-0', 'UKESM1-0-LL'], type=str)
    parser.add_argument('--topo_path', default="/lustre/home/eunhan/korea_downscaling_2km/merit_dem/MERIT_DEM_0p50deg.nc", type=str)
    parser.add_argument('--lmask_path', default='/lustre/home/eunhan/korea_downscaling_2km/land_sea_mask/land_sea_mask_0p50deg.nc', type=str)


    # dataset parameters
    parser.add_argument('--train_years', default=range(1961, 1990+1), type=list)
    parser.add_argument('--val_years', default=[1999], type=list)
    parser.add_argument('--test_years', default=range(2000,2015), type=list)

    parser.add_argument('--batch_size', '-b', default=8, type=int, help='Training batch size')
    parser.add_argument('--val_batch_size', '-vb', default=8, type=int, help='Validation batch size')
    parser.add_argument('--base_path', default=None, type=str)
    parser.add_argument('--mean_std_path', default=None, type=str)

    # method parameters
    parser.add_argument('--config_file', '-c', default=None, type=str,
                        help='Path to the default config file')

    # Training parameters (optimizer)
    parser.add_argument('--epochs', '-e', default=25, type=int)

    parser.add_argument('--ckpt_path', default=None, type=str)

    return parser
