import pandas as pd
import numpy as np
import os
import seaborn as sn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression as mi_reg
from sklearn.metrics import normalized_mutual_info_score as nmi


def create_dataset(data, folder='./images/', verbose=False):

    if not os.path.exists(folder):
        os.makedirs(folder)

    for c in data[37].keys():
        if verbose:
            print(c)
        dataset = pd.DataFrame()
        for type_v in data.keys():
            if c in data[type_v].keys():
                file_path = f'{data[type_v][c]}/features_coeffs.csv'
                features = pd.read_csv(file_path, index_col=0)
                features['vessel_type'] = np.repeat(type_v, features.shape[0])
                features = features.rename(columns={features.columns[0]: 'mmsi'})
                if 'port' in folder:
                    data_path = f'./data/Lubna/Labeled_trajectory_DCAIS_({type_v})_region_01-03_to_30-05_trips.csv'
                    raw_data = pd.read_csv(data_path, parse_dates=['time'], low_memory=False)
                    raw_data['time'] = raw_data['time'].astype('datetime64[ns]')
                    raw_data = raw_data[raw_data['label'] != 'navigating']
                    raw_data = raw_data.drop_duplicates('mmsi', keep='first')
                    raw_data = raw_data[['mmsi', 'label']]
                    features = pd.merge(features, raw_data, on='mmsi')
                elif 'navigating' in folder:
                    features['label'] = np.repeat('navigating', features.shape[0])
                else:
                    features['label'] = np.repeat('whole', features.shape[0])
                dataset = pd.concat([dataset, features], axis=0)

        dataset.columns = ['mmsi'] + list(dataset.columns[1:])
        if not os.path.exists(f'{folder}/{c}/'):
            os.makedirs(f'{folder}/{c}/')
        dataset.to_csv(f'{folder}/{c}/dataset.csv')


def plot_images(folder_path):
    folders = os.listdir(folder_path)

    for c in folders:
        data_path = f'{folder_path}/{c}'
        dataset = pd.read_csv(f'{data_path}/dataset.csv', index_col=0)
        # correlation
        corr_matrix = dataset.corr()
        plt.figure()
        sn.heatmap(corr_matrix, annot=True)
        plt.savefig(f'{data_path}/correlation.png')

        # MI
        dataset.drop(['mmsi', 'label'], axis=1, inplace=True)
        indep_vars = ['vessel_type']  # set independent vars
        dep_vars = dataset.columns.difference(indep_vars).tolist()
        df_mi = pd.DataFrame([mi_reg(dataset[indep_vars], dataset[dep_var]) for dep_var in dep_vars], index=dep_vars,
                             columns=indep_vars).apply(lambda x: x / x.max(), axis=0)
        plt.figure()
        sn.heatmap(df_mi, annot=True)
        plt.savefig(f'{data_path}/mi.png')

        # scatter
        plt.figure()
        sn.pairplot(dataset, hue='vessel_type', diag_kind = 'kde')
        plt.savefig(f'{data_path}/pairplot.png')


