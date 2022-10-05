from preprocessing.clean_trajectories import Trajectories
from approach.ar_models import Models1
from approach.clustering import Clustering
from datetime import datetime

# Number of vessels
n_samples = None
# Fishing type
vessel_type = [30]
# Time period
start_day = datetime(2020, 4, 1)
end_day = datetime(2020, 4, 7)
# Attributes
dim_set = ['lat', 'lon']
# Creating dataset
dataset = Trajectories(n_samples=n_samples, vessel_type=vessel_type, time_period=(start_day, end_day))


#main_folder = f'./results/DCAIS_example/'

dataset_dict = dataset.pandas_to_dict()


features1 = Models1(dataset=dataset_dict, features_opt='ou', dim_set=dim_set, folder='./results/DCAIS_example/')
features2 = Models1(dataset=dataset_dict, features_opt='arima', dim_set=dim_set, folder='./results/DCAIS_example/')
features3 = Models1(dataset=dataset_dict, features_opt='multi_arima', dim_set=dim_set, folder='./results/DCAIS_example/')
features4 = Models1(dataset=dataset_dict, features_opt='var', dim_set=dim_set, folder='./results/DCAIS_example/')

ashuashas
### Runing clustering
result_1 = Clustering(ais_data_path=dataset.preprocessed_path, dm=features.coeffs, cluster_algorithm='dbscan', folder=f'./results/DCAIS_example/', norm_dist=False)
result_2 = Clustering(ais_data_path=dataset.preprocessed_path, dm=features.coeffs, cluster_algorithm='hierarchical', linkage='average', folder=f'./results/DCAIS_example/', norm_dist=False)
result_3 = Clustering(ais_data_path=dataset.preprocessed_path, dm=features.coeffs, cluster_algorithm='spectral', folder=f'./results/DCAIS_example/', norm_dist=False)


