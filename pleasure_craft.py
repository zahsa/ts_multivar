from preprocessing.clean_trajectories import Trajectories
from approach.ar_models_cs import Models
from approach.ar_models import Models1
from approach.clustering import Clustering
from datetime import datetime
import os
import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score

# Number of vessels
n_samples = None
# Pleasure type
vessel_type = [37]
# Time period
start_day = datetime(2020, 4, 1)
end_day = datetime(2020, 6, 30)
# Attributes
dim_set = ['lat', 'lon']
# Creating dataset
dataset = Trajectories(n_samples=n_samples, vessel_type=vessel_type, time_period=(start_day, end_day))
dataset_dict = dataset.pandas_to_dict()

features_path_ou = f'./results/DCAIS_example/OU/features_coeffs.csv'
if not os.path.exists(features_path_ou):
    print('Univariate OU...')
    features_ou = Models1(dataset=dataset_dict, features_opt='ou', dim_set=dim_set, folder='./results/DCAIS_example/OU/')
    features_ou = features_ou.coeffs
else:
    print('ou coeffs exist')
    features_ou = pd.read_csv(features_path_ou, index_col=0)
features_path_arima = f'./results/DCAIS_example/ARIMA/features_coeffs.csv'
if not os.path.exists(features_path_arima):
    print('Univariate ARIMA...')
    features_arima = Models1(dataset=dataset_dict, features_opt='arima', dim_set=dim_set, folder='./results/DCAIS_example/ARIMA/')
    features_arima = features_arima.coeffs
else:
    print('arima coeffs exist')
    features_arima = pd.read_csv(features_path_arima, index_col=0)
features_path_var = f'./results/DCAIS_example/VAR/features_coeffs.csv'
if not os.path.exists(features_path_var):
    print('Univariate Multivariate...')
    features = Models(dataset=dataset_dict, features_opt='var', dim_set=dim_set, folder='./results/DCAIS_example/VAR/')
    features = features.coeffs
else:
    print('var coeffs exist')
    features = pd.read_csv(features_path_var, index_col=0)

### Runing clustering OU
print('Clustering OU...')

# result_ou_1 = Clustering(ais_data_path=dataset.preprocessed_path, dm=features_ou, cluster_algorithm='dbscan', folder=f'./results/DCAIS_example/OU/', norm_dist=False)

cl_path_ou_hc = './results/DCAIS_example/OU/hierarchical-average/hierarchical_5_average.csv'
if not os.path.exists(cl_path_ou_hc):
    result_ou_2 = Clustering(ais_data_path=dataset.preprocessed_path, dm=features_ou, cluster_algorithm='hierarchical', linkage='average', k=5, folder=f'./results/DCAIS_example/OU/', norm_dist=False)
    result_ou_2 = result_ou_2.labels
else:
    print('HC 5 exist')
    result_ou_2 = pd.read_csv(cl_path_ou_hc)['Clusters']

cl_path_ou_spectral = './results/DCAIS_example/OU/spectral/spectral_5.csv'
if not os.path.exists(cl_path_ou_spectral):
    result_ou_3 = Clustering(ais_data_path=dataset.preprocessed_path, dm=features_ou, cluster_algorithm='spectral', k=5, folder=f'./results/DCAIS_example/OU/', norm_dist=False)
    result_ou_3 = result_ou_3.labels
else:
    print('SC 5 exist')
    result_ou_2 = pd.read_csv(cl_path_ou_hc)['Clusters']

### Runing clustering ARIMA
print('Clustering ARIMA...')
# result_arima_1 = Clustering(ais_data_path=dataset.preprocessed_path, dm=features_arima, cluster_algorithm='dbscan', folder=f'./results/DCAIS_example/ARIMA/', norm_dist=False)

cl_path_arima_hc = './results/DCAIS_example/ARIMA/hierarchical-average/hierarchical_5_average.csv'
if not os.path.exists(cl_path_arima_hc):
    result_arima_2 = Clustering(ais_data_path=dataset.preprocessed_path, dm=features_arima, cluster_algorithm='hierarchical', linkage='average', k=5, folder=f'./results/DCAIS_example/ARIMA/', norm_dist=False)
    result_arima_2 = result_arima_2.labels
else:
    print('SC 5 exist')
    result_arima_2 = pd.read_csv(cl_path_ou_hc)['Clusters']

cl_path_arima_spectral = './results/DCAIS_example/ARIMA/spectral/spectral_5.csv'
if not os.path.exists(cl_path_arima_spectral):
    result_arima_3 = Clustering(ais_data_path=dataset.preprocessed_path, dm=features_arima, cluster_algorithm='spectral', k=5, folder=f'./results/DCAIS_example/ARIMA/', norm_dist=False)
    result_arima_3 = result_arima_3.labels
else:
    print('SC 5 exist')
    result_arima_3 = pd.read_csv(cl_path_ou_hc)['Clusters']

### Runing clustering
print('Clustering VAR...')
# result_1 = Clustering(ais_data_path=dataset.preprocessed_path, dm=features, cluster_algorithm='dbscan', folder=f'./results/DCAIS_example/', norm_dist=False)

cl_path_var_hc = './results/DCAIS_example/hierarchical-average/hierarchical_5_average.csv'
if not os.path.exists(cl_path_var_hc):
    result_2 = Clustering(ais_data_path=dataset.preprocessed_path, dm=features, cluster_algorithm='hierarchical', linkage='average', k=5, folder=f'./results/DCAIS_example/', norm_dist=False)
    result_2 = result_2.labels
else:
    print('SC 5 exist')
    result_2 = pd.read_csv(cl_path_ou_hc)['Clusters']

cl_path_var_spectral = './results/DCAIS_example/spectral/spectral_5.csv'
if not os.path.exists(cl_path_var_spectral):
    result_3 = Clustering(ais_data_path=dataset.preprocessed_path, dm=features, cluster_algorithm='spectral', k=5, folder=f'./results/DCAIS_example/', norm_dist=False)
    result_3 = result_3.labels
else:
    print('SC 5 exist')
    result_3 = pd.read_csv(cl_path_var_spectral)['Clusters']

#NMI
print('OU vs ARIMA')
# print(normalized_mutual_info_score(result_ou_1.labels, result_arima_1.labels))
print(normalized_mutual_info_score(result_ou_2, result_arima_2))
print(normalized_mutual_info_score(result_ou_3, result_arima_3))

print('ARIMA vs VAR')
# print(normalized_mutual_info_score(result_arima_1.labels, result_1.labels))
print(normalized_mutual_info_score(result_arima_2, result_2))
print(normalized_mutual_info_score(result_arima_3, result_3))

print('OU vs VAR')
# print(normalized_mutual_info_score(result_ou_1.labels, result_1.labels))
print(normalized_mutual_info_score(result_ou_2, result_2))
print(normalized_mutual_info_score(result_ou_3, result_3))