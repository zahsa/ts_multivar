import pandas as pd

from preprocessing import read_data as rd
from exp import AIC_models_exp as aicExp
import os
import classifiers.preparing_data as ppdt
from analysis import plot_images as pli

# Pleasure Craft, Passenger, Fishing, Tanker
vessel_list = [37, 60, 30, 80]
# Attributes
dim_set = ['lat', 'lon']
all_paths = {}

print('Whole trajectory')
for vessel_type in vessel_list:
    print(vessel_type)
    # 2020 - region_[47.5, 49.3, -125.5, -122.5]
    data_path = f'./data/DCAIS_[{vessel_type}]_region_[47.5, 49.3, -125.5, -122.5]_01-03_to_30-05_trips.csv'
    file_name = os.path.basename(data_path)
    file_name = os.path.splitext(file_name)[0]
    dataset_dict = rd.get_raw_dataset(data_path)

    main_folder = f'./results/all/{file_name}/'
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)

    all_paths[vessel_type] = aicExp.AIC_general(dataset_dict, dim_set, main_folder, ou_opt='SLSQP',
                arima_ar_p=3, arima_ma_p=0, var_ar_p=3, varmax_ar_p=3, varmax_ma_p=0)

pli.plot_all(all_paths, folder=f'./results/all/')
ppdt.create_dataset(all_paths, folder='./data/coeffs/')
ppdt.plot_images('./data/coeffs/')

print('Navigating')
for vessel_type in vessel_list:
    print(vessel_type)
    # 2020 - region_[47.5, 49.3, -125.5, -122.5]
    data_path = f'./data/Lubna/Labeled_trajectory_DCAIS_({vessel_type})_region_01-03_to_30-05_trips.csv'
    file_name = os.path.basename(data_path)
    file_name = os.path.splitext(file_name)[0]
    dataset_dict = rd.get_Lubna_dataset(data_path, navigation=True)

    main_folder = f'./results/all/{file_name}/navigating/'
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)

    all_paths[vessel_type] = aicExp.AIC_general(dataset_dict, dim_set, main_folder, ou_opt='SLSQP',
                arima_ar_p=3, arima_ma_p=0, var_ar_p=3, varmax_ar_p=3, varmax_ma_p=0)

pli.plot_all(all_paths, folder=f'./results/all/navigating/')
ppdt.create_dataset(all_paths, folder='./data/navigating/coeffs/')
ppdt.plot_images('./data/navigating/coeffs/')

print('Port')
for vessel_type in vessel_list:
    print(vessel_type)
    # 2020 - region_[47.5, 49.3, -125.5, -122.5]
    data_path = f'./data/Lubna/Labeled_trajectory_DCAIS_({vessel_type})_region_01-03_to_30-05_trips.csv'
    file_name = os.path.basename(data_path)
    file_name = os.path.splitext(file_name)[0]
    dataset_dict = rd.get_Lubna_dataset(data_path, navigation=False)

    main_folder = f'./results/all/{file_name}/port/'
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)

    all_paths[vessel_type] = aicExp.AIC_general(dataset_dict, dim_set, main_folder, ou_opt='SLSQP',
                arima_ar_p=3, arima_ma_p=0, var_ar_p=3, varmax_ar_p=3, varmax_ma_p=0)

pli.plot_all(all_paths, folder=f'./results/all/port/')
ppdt.create_dataset(all_paths, folder='./data/port/coeffs/')
ppdt.plot_images('./data/port/coeffs/')

# combining port and navigating datasets for second classification
print('Combining port and navigating')
folders_name = os.listdir("./data/port/coeffs/")
for filename in folders_name:
    data_n = pd.read_csv(f'./data/navigating/coeffs/{filename}/dataset.csv', index_col=0)
    data_n['labels2'] = 'navigating'
    data_p = pd.read_csv(f'./data/port/coeffs/{filename}/dataset.csv', index_col=0)
    data_p['labels2'] = 'port'
    join_data = pd.concat([data_n, data_p])
    if not os.path.exists(f'./data/join/coeffs/{filename}/'):
        os.makedirs(f'./data/join/coeffs/{filename}/')
    join_data.to_csv(f'./data/join/coeffs/{filename}/dataset.csv')
    print('\n')


# combining port and navigating datasets for second classification
print('Combining port and navigating for vessel type')
folders_name = os.listdir("./data/port/coeffs/")
for filename in folders_name:
    data_n = pd.read_csv(f'./data/navigating/coeffs/{filename}/dataset.csv', index_col=0)
    data_p = pd.read_csv(f'./data/port/coeffs/{filename}/dataset.csv', index_col=0)
    data_p = data_p.drop(['vessel_type', 'label'], axis=1)
    join_data = pd.merge(data_n, data_p, on='mmsi')
    if not os.path.exists(f'./data/join_2/coeffs/{filename}/'):
        os.makedirs(f'./data/join_2/coeffs/{filename}/')
    join_data.to_csv(f'./data/join_2/coeffs/{filename}/dataset.csv')
    print('\n')


# combining var port and ou navigating datasets for second classification
print('Combining port and navigating for vessel type')
data_n = pd.read_csv(f'./data/navigating/coeffs/ou-SLSQP/dataset.csv', index_col=0)
data_p = pd.read_csv(f'./data/port/coeffs/var_3_n/dataset.csv', index_col=0)
data_p = data_p.drop(['vessel_type', 'label'], axis=1)
join_data = pd.merge(data_n, data_p, on='mmsi')
if not os.path.exists(f'./data/join_3/coeffs/ou-var/'):
    os.makedirs(f'./data/join_3/coeffs/ou-var/')
join_data.to_csv(f'./data/join_3/coeffs/ou-var/dataset.csv')
print('\n')
