from preprocessing import read_data as rd
from exp import AIC_models_exp as aicExp
import os
from analysis import plot_images as pli
import pandas as pd

# Fishing type
vessel_list = [37, 60, 30, 80]
# Attributes
dim_set = ['lat', 'lon']
all_paths = {}

#navigation
print("\nNAVIGATION EXECUTION \n")
for vessel_type in vessel_list:
    print(vessel_type)
    # 2020 - region_[47.5, 49.3, -125.5, -122.5]
    data_path = f'./data/Lubna/Labeled_trajectory_DCAIS_({vessel_type})_region_01-03_to_30-05_trips.csv'
    file_name = os.path.basename(data_path)
    file_name = os.path.splitext(file_name)[0]
    dataset_dict = rd.get_Lubna_dataset(data_path, navigation=True, samples=100)

    main_folder = f'./results/{file_name}/navigating/'
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)

    all_paths[vessel_type] = aicExp.AIC_experiments(dataset_dict, dim_set, main_folder)
    # all_paths[vessel_type] = aicExp.AIC_general(dataset_dict, dim_set, main_folder)

pli.plot_all(all_paths, folder=f'./results/navigating/')
print('AIC', end=" ")
for c in all_paths[37].keys():
    print(c)
    for type_v in all_paths.keys():
        if c in all_paths[type_v].keys():
            file_path = f'{all_paths[type_v][c]}/features_measures.csv'
            info_data = pd.read_csv(file_path)
            print(f'& {round(info_data.AIC.median())}', end=" ")
        else:
            print(f'& - ', end=" ")
    print('\\')

#port
print("\nPORT EXECUTION \n")
for vessel_type in vessel_list:
    print(vessel_type)
    # 2020 - region_[47.5, 49.3, -125.5, -122.5]
    data_path = f'./data/Lubna/Labeled_trajectory_DCAIS_({vessel_type})_region_01-03_to_30-05_trips.csv'
    file_name = os.path.basename(data_path)
    file_name = os.path.splitext(file_name)[0]
    dataset_dict = rd.get_Lubna_dataset(data_path, navigation=False, samples=100)

    main_folder = f'./results/{file_name}/port/'
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)

    all_paths[vessel_type] = aicExp.AIC_experiments(dataset_dict, dim_set, main_folder)
    # all_paths[vessel_type] = aicExp.AIC_general(dataset_dict, dim_set, main_folder)

pli.plot_all(all_paths, folder=f'./results/port/')
print('AIC', end=" ")
for c in all_paths[37].keys():
    print(c)
    for type_v in all_paths.keys():
        if c in all_paths[type_v].keys():
            file_path = f'{all_paths[type_v][c]}/features_measures.csv'
            info_data = pd.read_csv(file_path)
            print(f'& {round(info_data.AIC.median())}', end=" ")
        else:
            print(f'& - ', end=" ")
    print('\\')


