import numpy as np
import os
from datetime import datetime, timedelta
from FWIFunctions_v6 import *

from netCDF4 import Dataset

def convert_fwi_units(temperature_k, humidity_kgkg, wind_ms, pressure_hPa=1013.25):
    """
    Convert Fire Weather Index (FWI) related data units.

    Parameters:
        temperature_k (float or np.array): Temperature in Kelvin
        humidity_kgkg (float or np.array): Specific humidity in kg/kg
        wind_ms (float or np.array): Wind speed in m/s
        rain_kgm2s (float or np.array): Rainfall rate in kg/m²/s

    Returns:
        dict: Converted values in Celsius, %, km/h, and mm/h
    """
    temperature_c = temperature_k - 273.15  # Kelvin to Celsius
    # temperature_c = np.clip(temperature_c, -50, 50)
    
    e = (humidity_kgkg * pressure_hPa) / (0.622 + (1 - 0.622) * humidity_kgkg)
    
    # Calculate the saturation vapor pressure (e_s) in hPa
    e_s = 6.112 * np.exp((17.67 * temperature_c) / (temperature_c + 243.5))
    
    # Calculate relative humidity (RH) in %
    humidity_percent = (e / e_s) * 100
    humidity_percent = np.clip(humidity_percent, 0, 100)
    wind_kph = wind_ms * 3.6  # m/s to km/h

    return temperature_c, humidity_percent, wind_kph

def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

if __name__ == '__main__':
    input_dir = "/lustre/home/eunhan/korea_downscaling_2km/"
    result_path = "./result/LR/"
    years = range(2000, 2014+1)
    
    GCMs_list = ['GFDL-ESM4', 'IPSL-CM6A-LR', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'UKESM1-0-LL']
    data_types = ["historical", "picontrol"]

    lat = np.load("../data/ISIMIP_ko_lat_0p50deg.npy")
    lon = np.load("../data/ISIMIP_ko_lon_0p50deg.npy")
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    for year in years:
        start_date = datetime(year, 1, 1)
        days_in_year = 366 if is_leap_year(year) else 365  # 윤년 여부 체크
        datetimes = [start_date + timedelta(days=i) for i in range(days_in_year)]

        for data_type_ in data_types:
            for GCM in GCMs_list:
                input_nc = Dataset(os.path.join(input_dir, f"{data_type_}2", GCM, f'{year}.nc'), 'r', format = 'netcdf4')

                humidity_kgkg = np.array(input_nc['huss'])
                rain_npy = np.array(input_nc['pr']) * 86400
                press_npy = np.array(input_nc['ps'])*0.01
                wind_ms = np.array(input_nc['sfcwind'])
                temperature_k = np.array(input_nc['tasmax'])
                temp_npy, humi_npy, wind_npy = convert_fwi_units(temperature_k, humidity_kgkg, wind_ms, press_npy)

                gcm_result_path = os.path.join(result_path, data_type_, GCM)
                os.makedirs(gcm_result_path, exist_ok=True)

                # 결과 배열 초기화 (shape: 365 x 25 x 25)
                fwi_result = np.zeros((days_in_year, 25, 25), dtype=np.float64)
                ffmc_result = np.zeros_like(fwi_result)
                dmc_result = np.zeros_like(fwi_result)
                dc_result = np.zeros_like(fwi_result)
                isi_result = np.zeros_like(fwi_result)
                bui_result = np.zeros_like(fwi_result)

                init_ffmc = np.full((25, 25), 85, dtype=np.float64)
                init_dmc = np.full((25, 25), 6, dtype=np.float64)
                init_dc = np.full((25, 25), 15, dtype=np.float64)

                # Fire Weather Index 계산 루프
                for idx, date in enumerate(datetimes):
                    file_name = os.path.join(gcm_result_path, f"{date.strftime("%Y%m%d")}.npy")

                    if idx == 0:
                        ffmc = FFMC(temp_npy[idx], humi_npy[idx], wind_npy[idx], rain_npy[idx], init_ffmc)
                        dmc = DMC(temp_npy[idx], humi_npy[idx], rain_npy[idx], init_dmc, lat_grid, date.day, date.month)
                        dc = DC(temp_npy[idx], rain_npy[idx], init_dc, lat_grid, date.month)
                        isi = ISI(wind_npy[idx], ffmc)
                        bui = BUI(dmc, dc)
                        fwi = FWI(isi, bui)
                    else:
                        ffmc = FFMC(temp_npy[idx], humi_npy[idx], wind_npy[idx], rain_npy[idx], ffmc_result[idx - 1])
                        dmc = DMC(temp_npy[idx], humi_npy[idx], rain_npy[idx], dmc_result[idx - 1], lat_grid, date.day, date.month)
                        dc = DC(temp_npy[idx], rain_npy[idx], dc_result[idx - 1], lat_grid, date.month)
                        isi = ISI(wind_npy[idx], ffmc)
                        bui = BUI(dmc, dc)
                        fwi = FWI(isi, bui)

                    # 결과 저장
                    fwi_result[idx] = fwi
                    ffmc_result[idx] = ffmc
                    dmc_result[idx] = dmc
                    dc_result[idx] = dc
                    isi_result[idx] = isi
                    bui_result[idx] = bui

                    np.save(file_name, fwi_result[idx])
                    # print(idx)

                print(f"Processing complete. Results saved to {gcm_result_path}")
