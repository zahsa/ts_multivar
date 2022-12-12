from preprocessing.clean_trajectories import Trajectories
from ARmodels.ar_models_cs import Models
from ARmodels.ar_models import Models1
from ARmodels.clustering import Clustering
from datetime import datetime
import os
import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import stumpy

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
# dataset_dict = dataset.pandas_to_dict()

features_path_ou = f'./results/DCAIS_example/OU/features_coeffs.csv'
if not os.path.exists(features_path_ou):
    print('Univariate OU...')
    dataset_dict = dataset.pandas_to_dict()
    features_ou = Models1(dataset=dataset_dict, features_opt='ou', dim_set=dim_set, folder='./results/DCAIS_example/OU/')
    features_ou = features_ou.coeffs
else:
    print('ou coeffs exist')
    features_ou = pd.read_csv(features_path_ou, index_col=0)
features_path_arima = f'results/DCAIS_example/ARIMA/features_coeffs.csv'
if not os.path.exists(features_path_arima):
    print('Univariate ARIMA...')
    dataset_dict = dataset.pandas_to_dict()
    features_arima = Models1(dataset=dataset_dict, features_opt='arima', dim_set=dim_set, folder='./results/DCAIS_example/ARIMA/')
    features_arima = features_arima.coeffs
else:
    print('arima coeffs exist')
    features_arima = pd.read_csv(features_path_arima, index_col=0)
features_path_var = f'./results/DCAIS_example/VAR/features_coeffs.csv'
if not os.path.exists(features_path_var):
    print('Univariate Multivariate...')
    dataset_dict = dataset.pandas_to_dict()
    features = Models(dataset=dataset_dict, features_opt='var', dim_set=dim_set, folder='./results/DCAIS_example/VAR/')
    features = features.coeffs
else:
    print('var coeffs exist')
    features = pd.read_csv(features_path_var, index_col=0)

### Runing clustering OU
print('Clustering OU...')
result_ou_1 = Clustering(ais_data_path=dataset.preprocessed_path, dm=features_ou, cluster_algorithm='dbscan', folder=f'./results/DCAIS_example/OU/', norm_dist=False)
print(f'\t Silhouette: {result_ou_1.Cl_Silhouette[0]}')
result_ou_1 = result_ou_1.labels

cl_path_ou_hc = './results/DCAIS_example/OU/hierarchical-average/hierarchical_5_average.csv'
if not os.path.exists(cl_path_ou_hc):
    result_ou_2 = Clustering(ais_data_path=dataset.preprocessed_path, dm=features_ou, cluster_algorithm='hierarchical', linkage='average', k=5, folder=f'./results/DCAIS_example/OU/', norm_dist=False)
    print(f'\t Silhouette: {result_ou_2.Cl_Silhouette[0]}')
    result_ou_2 = result_ou_2.labels
else:
    print('HC 5 exist')
    result_ou_2 = pd.read_csv(cl_path_ou_hc)
    print(f'\t Silhouette: {result_ou_2.Cl_Silhouette[0]}')
    result_ou_2 = result_ou_2.groupby('trips').first()['Clusters']
print(result_ou_2.labels)

cl_path_ou_spectral = './results/DCAIS_example/OU/spectral/spectral_5.csv'
if not os.path.exists(cl_path_ou_spectral):
    result_ou_3 = Clustering(ais_data_path=dataset.preprocessed_path, dm=features_ou, cluster_algorithm='spectral', k=5, folder=f'./results/DCAIS_example/OU/', norm_dist=False)
    print(f'\t Silhouette: {result_ou_3.Cl_Silhouette[0]}')
    result_ou_3 = result_ou_3.labels
else:
    print('SC 5 exist')
    result_ou_3 = pd.read_csv(cl_path_ou_spectral)
    print(f'\t Silhouette: {result_ou_3.Cl_Silhouette[0]}')
    result_ou_3 = result_ou_3.groupby('trips').first()['Clusters']

result_ou_4 = Clustering(ais_data_path=dataset.preprocessed_path, dm=features_ou, cluster_algorithm='hdbscan', folder=f'./results/DCAIS_example/OU/', norm_dist=False)
print(f'\t Silhouette: {result_ou_4.Cl_Silhouette[0]}')
result_ou_4 = result_ou_4.labels

### Runing clustering ARIMA
print('Clustering ARIMA...')
result_arima_1 = Clustering(ais_data_path=dataset.preprocessed_path, dm=features_arima, cluster_algorithm='dbscan', folder=f'./results/DCAIS_example/ARIMA/', norm_dist=False)
print(f'\t Silhouette: {result_arima_1.Cl_Silhouette[0]}')
result_arima_1 = result_arima_1.labels

cl_path_arima_hc = 'results/DCAIS_example/ARIMA/hierarchical-average/hierarchical_5_average.csv'
if not os.path.exists(cl_path_arima_hc):
    result_arima_2 = Clustering(ais_data_path=dataset.preprocessed_path, dm=features_arima, cluster_algorithm='hierarchical', linkage='average', k=5, folder=f'./results/DCAIS_example/ARIMA/', norm_dist=False)
    print(f'\t Silhouette: {result_arima_2.Cl_Silhouette[0]}')
    result_arima_2 = result_arima_2.labels
else:
    print('HC 5 exist')
    result_arima_2 = pd.read_csv(cl_path_arima_hc)
    print(f'\t Silhouette: {result_arima_2.Cl_Silhouette[0]}')
    result_arima_2 = result_arima_2.groupby('trips').first()['Clusters']

cl_path_arima_spectral = 'results/DCAIS_example/ARIMA/spectral/spectral_5.csv'
if not os.path.exists(cl_path_arima_spectral):
    result_arima_3 = Clustering(ais_data_path=dataset.preprocessed_path, dm=features_arima, cluster_algorithm='spectral', k=5, folder=f'./results/DCAIS_example/ARIMA/', norm_dist=False)
    print(f'\t Silhouette: {result_arima_3.Cl_Silhouette[0]}')
    result_arima_3 = result_arima_3.labels
else:
    print('SC 5 exist')
    result_arima_3 = pd.read_csv(cl_path_arima_spectral)
    print(f'\t Silhouette: {result_arima_3.Cl_Silhouette[0]}')
    result_arima_3 = result_arima_3.groupby('trips').first()['Clusters']

result_arima_4 = Clustering(ais_data_path=dataset.preprocessed_path, dm=features_arima, cluster_algorithm='hdbscan', folder=f'./results/DCAIS_example/ARIMA/', norm_dist=False)
print(f'\t Silhouette: {result_arima_4.Cl_Silhouette[0]}')
result_arima_4 = result_arima_4.labels

### Runing clustering
print('Clustering VAR...')
result_1 = Clustering(ais_data_path=dataset.preprocessed_path, dm=features, cluster_algorithm='dbscan', folder=f'./results/DCAIS_example/VAR/', norm_dist=False)
print(f'\t Silhouette: {result_1.Cl_Silhouette[0]}')
result_1 = result_1.labels

cl_path_var_hc = './results/DCAIS_example/VAR/hierarchical-average/hierarchical_5_average.csv'
if not os.path.exists(cl_path_var_hc):
    result_2 = Clustering(ais_data_path=dataset.preprocessed_path, dm=features, cluster_algorithm='hierarchical', linkage='average', k=5, folder=f'./results/DCAIS_example/VAR/', norm_dist=False)
    print(f'\t Silhouette: {result_2.Cl_Silhouette[0]}')
    result_2 = result_2.labels
else:
    print('HC 5 exist')
    result_2 = pd.read_csv(cl_path_var_hc)
    print(f'\t Silhouette: {result_2.Cl_Silhouette[0]}')
    result_2 = result_2.groupby('trips').first()['Clusters']

cl_path_var_spectral = './results/DCAIS_example/VAR/spectral/spectral_5.csv'
if not os.path.exists(cl_path_var_spectral):
    result_3 = Clustering(ais_data_path=dataset.preprocessed_path, dm=features, cluster_algorithm='spectral', k=5, folder=f'./results/DCAIS_example/VAR/', norm_dist=False)
    print(f'\t Silhouette: {result_3.Cl_Silhouette[0]}')
    result_3 = result_3.labels
else:
    print('SC 5 exist')
    result_3 = pd.read_csv(cl_path_var_spectral)
    print(f'\t Silhouette: {result_3.Cl_Silhouette[0]}')
    result_3 = result_3.groupby('trips').first()['Clusters']

result_4 = Clustering(ais_data_path=dataset.preprocessed_path, dm=features, cluster_algorithm='hdbscan', folder=f'./results/DCAIS_example/VAR/', norm_dist=False)
print(f'\t Silhouette: {result_4.Cl_Silhouette[0]}')
result_4 = result_4.labels



#NMI
print('OU vs ARIMA')
print(adjusted_rand_score(result_ou_1, result_arima_1))
print(adjusted_rand_score(result_ou_2, result_arima_2))
print(adjusted_rand_score(result_ou_3, result_arima_3))

print('ARIMA vs VAR')
print(adjusted_rand_score(result_arima_1, result_1))
print(adjusted_rand_score(result_arima_2, result_2))
print(adjusted_rand_score(result_arima_3, result_3))

print('OU vs VAR')
print(adjusted_rand_score(result_ou_1, result_1))
print(adjusted_rand_score(result_ou_2, result_2))
print(adjusted_rand_score(result_ou_3, result_3))
