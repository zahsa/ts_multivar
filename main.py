from preprocessing import read_data as rd
from exp import AIC_models_exp as aicExp
import os
import classifiers.preparing_data as ppdt
from analysis import plot_images as pli
from classifiers import MLmodels as ml

# Fishing type
vessel_list = [37, 60, 30, 80]
# Attributes
dim_set = ['lat', 'lon']
all_paths = {}

# for vessel_type in vessel_list:
#     print(vessel_type)
#     # 2021 - region_[47.5, 49.3, -125.5, -122.5]
#     data_path = f'./data/DCAIS_[{vessel_type}]_region_[47.5, 49.3, -125.5, -122.5]_01-03_to_30-05_trips.csv'
#     file_name = os.path.basename(data_path)
#     file_name = os.path.splitext(file_name)[0]
#     dataset_dict = rd.get_raw_dataset(data_path)
#
#     main_folder = f'./results/all/{file_name}/'
#     if not os.path.exists(main_folder):
#         os.makedirs(main_folder)
#
#     # all_paths[vessel_type] = aicExp.AIC_experiments(dataset_dict, dim_set, main_folder)
#     all_paths[vessel_type] = aicExp.AIC_general(dataset_dict, dim_set, main_folder)

# pli.plot_all(all_paths, folder=f'./results/')
# ppdt.create_dataset(all_paths, folder='./data/coeffs/')
# ppdt.plot_images('./data/coeffs/')
ml.apply_ML('./data/coeffs/ou-L-BFGS-B/')
ml.apply_ML('./data/coeffs/arima_3_0_n/')
# ml.apply_ML('./data/coeffs/ou-SLSQP/')
# ml.apply_ML('./data/coeffs/var_2_n/')


