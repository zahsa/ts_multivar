from preprocessing.clean_trajectories import Trajectories
from approach.ar_models_cs import Models
from approach.clustering import Clustering
from datetime import datetime

# Number of vessels
n_samples = None
# Fishing type
vessel_type = [30, 1001, 1002]
# Time period
start_day = datetime(2020, 4, 1)
end_day = datetime(2020, 4, 30)
# Attributes
dim_set = ['lat', 'lon']
# Creating dataset
dataset = Trajectories(n_samples=n_samples, vessel_type=vessel_type, time_period=(start_day, end_day))


#main_folder = f'./results/DCAIS_example/'

dataset_dict = dataset.pandas_to_dict()


features = Models(dataset=dataset_dict, features_opt='var', dim_set=dim_set, folder='./results/DCAIS_example/')


