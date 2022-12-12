from preprocessing import read_data as rd
from ARmodels.ar_models import Models1
from analysis import plot_images as pli
import pandas as pd
import os

# Fishing type
vessel_type = [80]
# Attributes
dim_set = ['lat', 'lon']
measure = 'AIC'

# 2021
data_path = f'./data/DCAIS_{vessel_type}_region_[47.5, 49.3, -125.5, -122.5]_01-03_to_30-05_trips.csv'
file_name = os.path.basename(data_path)
file_name = os.path.splitext(file_name)[0]
dataset_dict = rd.get_raw_dataset(data_path, samples=100)
tr = 'n'

# univariate
# features1 = Models1(dataset=dataset_dict, features_opt='ou', dim_set=dim_set, folder='./results/DCAIS_example/')
# features2 = Models1(dataset=dataset_dict, features_opt='arima', dim_set=dim_set, folder='./results/DCAIS_example/')
#
# #exog features
# features3 = Models1(dataset=dataset_dict, features_opt='multi_arima', dim_set=dim_set, folder='./results/DCAIS_example/')
#
# #multivariate
# features4 = Models1(dataset=dataset_dict, features_opt='var', dim_set=dim_set, folder='./results/DCAIS_example/')
# features5 = Models1(dataset=dataset_dict, features_opt='varmax', dim_set=dim_set, folder='./results/DCAIS_example/')

main_folder = f'./results/{file_name}/'
if not os.path.exists(main_folder):
    os.makedirs(main_folder)
#
curr_config_ou = {}
curr_config = {}
features = Models1(dataset=dataset_dict, features_opt='ou', dim_set=dim_set, folder=main_folder)
curr_config_ou['ou'] = features.path
curr_config['ou'] = features.path
pli.info_to_plot(curr_config_ou, model='ou', measure=measure, folder=main_folder)

print('ARIMA')
curr_config_arima = {}
for ar_p in [1, 2, 3]:
    for ma_p in [0, 1, 2, 3]:
        # for tr in ['c', 'n']:
        features = Models1(dataset=dataset_dict, features_opt='arima', dim_set=dim_set, ar_prm=ar_p, ma_prm=ma_p, trend=tr, folder=main_folder)
        curr_config_arima[f'{ar_p}_{ma_p}_{tr}'] = features.path
        curr_config[f'arima_{ar_p}_{ma_p}_{tr}'] = features.path
pli.info_to_plot(curr_config_arima, model='arima', measure=measure, folder=main_folder)

print('VAR')
curr_config_var = {}
for ar_p in [1, 2, 3]:
    # for tr in ['n', 'c']:
    features = Models1(dataset=dataset_dict, features_opt='var', dim_set=dim_set, ar_prm=ar_p, ma_prm=0, trend=tr, folder=main_folder)
    curr_config_var[f'{ar_p}_{tr}'] = features.path
    curr_config[f'var_{ar_p}_{tr}'] = features.path
pli.info_to_plot(curr_config_var, model='var', measure=measure, folder=main_folder)

print('MULTIARIMA')
curr_config_multiarima = {}
for ar_p in [1, 2, 3]:
    for ma_p in [0, 1, 2, 3]:
        # for tr in ['n','c']:
        features = Models1(dataset=dataset_dict, features_opt='multi_arima', dim_set=dim_set, ar_prm=ar_p, ma_prm=ma_p, trend=tr, folder=main_folder)
        curr_config_multiarima[f'{ar_p}_{ma_p}_{tr}'] = features.path
        curr_config[f'multi_arima_{ar_p}_{ma_p}_{tr}'] = features.path

pli.info_to_plot(curr_config_multiarima, model='multi_arima', measure=measure, folder=main_folder)

print('VARMAX')
curr_config_varma = {}
for ar_p in [1, 2, 3]:
    for ma_p in [0, 1, 2, 3]:
        # for tr in ['n', 'c']:
        features = Models1(dataset=dataset_dict, features_opt='varmax', dim_set=dim_set, ar_prm=ar_p, ma_prm=ma_p, trend=tr, folder=main_folder)
        curr_config_varma[f'{ar_p}_{ma_p}_{tr}'] = features.path
        curr_config[f'varma_{ar_p}_{ma_p}_{tr}'] = features.path
pli.info_to_plot(curr_config_varma, model='varmax', measure=measure, folder=main_folder)

pli.info_to_plot(curr_config, measure=measure, folder=main_folder)


