from preprocessing import read_data as rd
from approach.ar_models import Models1
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import os
import plotly.graph_objects as go


def boxplot_model(data, measure='AIC', model='arima', folder='./images/'):
    dim = 'Latitude and Longitude'

    x = pd.DataFrame()
    for i in data.keys():
        data[i][measure] = data[i][measure][data[i][measure] < 1e4]
        if f'{measure}.1' in data[i].keys():
            data[i][f'{measure}.1'] = data[i][f'{measure}.1'][data[i][f'{measure}.1'] < 1e4]
            x = pd.concat([x, (data[i][measure]+data[i][f'{measure}.1'])/2], axis=1)
        else:
            x = pd.concat([x, data[i][measure]], axis=1)

    x.columns = data.keys()
    x = x.replace([np.inf], np.nan)
    x = x.replace([np.nan], x.max().max())
    x = x.iloc[0:100, :]
    print(x.shape)

    # Plot
    fig = go.Figure()
    for i in x.columns:
        fig.add_trace(go.Violin(y=x[i],
                            name=i))
    fig.update_traces(box_visible=True, meanline_visible=True, showlegend=False)
    fig.update_layout(title_text=model, width=1000, height=700)
    # fig.show()
    fig.write_image(f'{folder}/features_{measure}_{dim}_{model}.png', scale=1)

    table = pd.concat([x.mean(), x.std(), x.max(), x.min(), x.quantile(0.25), x.median(), x.quantile(0.75)], axis=1)
    table.to_csv(f'{folder}/{measure}_{dim}_stats.csv')


def boxplot_all(data, measure='AIC', folder='./images/'):
    dim = 'Latitude and Longitude'

    x = pd.DataFrame()
    col = {}

    for i in data.keys():
        data[i][measure] = data[i][measure][data[i][measure] < 1e4]
        if f'{measure}.1' in data[i].keys():
            data[i][f'{measure}.1'] = data[i][f'{measure}.1'][data[i][f'{measure}.1'] < 1e4]
            x = pd.concat([x, (data[i][measure]+data[i][f'{measure}.1'])/2], axis=1)
            if 'ou' in i:
                col[i] = 'orange'
            elif 'multi_arima' in i:
                col[i] = 'blue'
            elif 'arima' in i:
                col[i] = 'green'
        else:
            x = pd.concat([x, data[i][measure]], axis=1)
            if 'varma' in i:
                col[i] = 'black'
            elif 'var' in i:
                col[i] = 'red'

    x.columns = data.keys()
    x = x.replace([np.inf], np.nan)
    x = x.replace([np.nan], x.max().max())
    x = x.iloc[0:100, :]
    print(x.shape)
    # Plot

    fig = go.Figure()
    for i in x.columns:
        fig.add_trace(go.Violin(y=x[i],
                            name=i,
                            line_color=col[i]))
    fig.update_traces(box_visible=True, meanline_visible=True)
    fig.update_layout(width=1300, height=1000)
    # fig.show()
    fig.write_image(f'{folder}/features_{measure}_{dim}_all.png', scale=3)

    table = pd.concat([x.mean(), x.std(), x.max(), x.min(), x.quantile(0.25), x.median(), x.quantile(0.75)], axis=1)
    table.columns = ['mean', 'std', 'max', 'min', 'quart25', 'median', 'quart75']
    table.to_csv(f'{folder}/{measure}_{dim}_stats.csv')


# Fishing type
vessel_type = [80]
# Attributes
dim_set = ['lat', 'lon']
measure = 'AIC'

# 2021
data_path = f'./data/DCAIS_{vessel_type}_region_[47.5, 49.3, -125.5, -122.5]_01-03_to_30-05_trips.csv'
file_name = os.path.basename(data_path)
file_name = os.path.splitext(file_name)[0]
dataset_dict = rd.get_raw_dataset(data_path, samples=300)

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
file_path = f'{features.path}/features_measures.csv'
curr_config_ou['ou'] = pd.read_csv(file_path)
curr_config['ou'] = pd.read_csv(file_path)
boxplot_model(curr_config_ou, model='ou', measure=measure, folder=main_folder)

print('ARIMA')
curr_config_arima = {}
for ar_p in [1, 2, 3]:
    for ma_p in [0, 1, 2, 3]:
        for tr in ['c', 'n']:
            features = Models1(dataset=dataset_dict, features_opt='arima', dim_set=dim_set, ar_prm=ar_p, ma_prm=ma_p, trend=tr, folder=main_folder)
            file_path = f'{features.path}/features_measures.csv'
            curr_config_arima[f'{ar_p}_{ma_p}_{tr}'] = pd.read_csv(file_path)
            curr_config[f'arima_{ar_p}_{ma_p}_{tr}'] = pd.read_csv(file_path)
boxplot_model(curr_config_arima, model='arima', measure=measure, folder=main_folder)

print('VAR')
curr_config_var = {}
for ar_p in [1, 2, 3]:
    for tr in ['n', 'c']:
        try:
            features = Models1(dataset=dataset_dict, features_opt='var', dim_set=dim_set, ar_prm=ar_p, ma_prm=0, trend=tr, folder=main_folder)
            file_path = f'{features.path}/features_measures.csv'
            curr_config_var[f'{ar_p}_{tr}'] = pd.read_csv(file_path)
            curr_config[f'var_{ar_p}_{tr}'] = pd.read_csv(file_path)
        except:
            print(f'Error when running: var_{ar_p}_{tr}')
boxplot_model(curr_config_var, model='var', measure=measure, folder=main_folder)

print('MULTIARIMA')
curr_config_multiarima = {}
for ar_p in [1, 2, 3]:
    for ma_p in [0, 1, 2, 3]:
        for tr in ['n', 'c']:
            try:
                features = Models1(dataset=dataset_dict, features_opt='multi_arima', dim_set=dim_set, ar_prm=ar_p, ma_prm=ma_p, trend=tr, folder=main_folder)
                file_path = f'{features.path}/features_measures.csv'
                curr_config_multiarima[f'{ar_p}_{ma_p}_{tr}'] = pd.read_csv(file_path)
                curr_config[f'multi_arima_{ar_p}_{ma_p}_{tr}'] = pd.read_csv(file_path)
            except:
                print(f'Error when running: multi_arima_{ar_p}_{ma_p}_{tr}')
boxplot_model(curr_config_multiarima, model='multi_arima', measure=measure, folder=main_folder)

#TODO: still requires tets
print('VARMAX')
curr_config_varma = {}
for ar_p in [1, 2]:
    for ma_p in [1, 2]:
        for tr in ['n', 'c']:
            try:
                features = Models1(dataset=dataset_dict, features_opt='varmax', dim_set=dim_set, ar_prm=ar_p, ma_prm=ma_p, trend=tr, folder=main_folder)
                file_path = f'{features.path}/features_measures.csv'
                curr_config_varma[f'{ar_p}_{ma_p}_{tr}'] = pd.read_csv(file_path)
                curr_config[f'varma_{ar_p}_{ma_p}_{tr}'] = pd.read_csv(file_path)
            except:
                print(f'Error when running: multi_arima_{ar_p}_{ma_p}_{tr}')
boxplot_model(curr_config_varma, model='varmax', measure=measure, folder=main_folder)

boxplot_all(curr_config, measure=measure, folder=main_folder)


