import os
import numpy as np
import pickle
from netCDF4 import Dataset

# 경로 및 변수 설정
path = "/lustre/home/eunhan/korea_downscaling_2km/picontrol2/"
# 최종 결과 저장
output_path = "./data"
os.makedirs(output_path, exist_ok=True)
output_file = "ISIMIP3b_picontrol2_1961_1990_ko_mean_std.pickle"
years = range(1961, 1990 + 1)
file_list = [f'{year}.nc' for year in years]
vals_list = ['tasmax', 'huss', 'ps', 'pr', 'rlds', 'rsds', 'sfcwind']
GCMs_list = ['GFDL-ESM4', 'IPSL-CM6A-LR', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'UKESM1-0-LL']

# 결과 저장 딕셔너리
GCMs_dict_result = {}

for gcm in GCMs_list:
    print(f"Processing {gcm}...")  # 진행 상황 출력
    input_dir = os.path.join(path, gcm)

    # 모델별 데이터 저장용 딕셔너리
    model_data = {var: {'mean': None, 'std': None} for var in vals_list}

    # 변수별 데이터 저장 딕셔너리
    var_data_dict = {var: [] for var in vals_list}

    # 해당 모델의 모든 파일 순회
    for file_name in file_list:
        file_path = os.path.join(input_dir, file_name)

        try:
            with Dataset(file_path, 'r') as nc_data:
                for idx in range(len(vals_list)):
                    val = vals_list[idx]
                    if val in nc_data.variables:  # 변수가 실제 존재하는지 확인
                        npy_data = np.array(nc_data.variables[val][:], dtype=np.float32)  # float32로 변환
                        if idx == 1: # huss
                            npy_data = np.log(1 + npy_data * 1000)
                        elif idx == 2: # ps
                            npy_data = np.log(npy_data / 100)
                        elif idx == 3: # pr
                            npy_data = np.log(1 + npy_data * 86400)
                        elif idx == 4: # rlds
                            npy_data = np.log(npy_data)
                        elif idx == 5: # rsds
                            npy_data = np.log(1 + npy_data)
                        var_data_dict[val].append(npy_data)  # 리스트에 데이터 추가

        except Exception as e:
            print(f"파일 {file_path} 읽기 실패: {e}")

    # 변수별 평균 및 표준편차 계산
    for var in vals_list:
        if var_data_dict[var]:  # 데이터가 존재하는 경우만 처리
            var_data = np.concatenate(var_data_dict[var], axis=0)  # 모든 연도 데이터 합치기
            model_data[var]['mean'] = np.nanmean(var_data, axis=0)  # (lat, lon) 형태 유지
            model_data[var]['std'] = np.nanstd(var_data, axis=0)  # (lat, lon) 형태 유지

    # 모델별 결과 저장
    GCMs_dict_result[gcm] = model_data

with open(os.path.join(output_path, output_file), 'wb') as fw:
    pickle.dump(GCMs_dict_result, fw)

print(f"Processing complete. Results saved to {os.path.join(output_path, output_file)}")
