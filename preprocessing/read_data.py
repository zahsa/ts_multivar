import pandas as pd
import numpy as np


def pandas_to_dict(dataset):
    """
    It converts the pandas dataset into dict format.
    :return: dataset in a pandas format.
    """
    new_dataset = {}
    ids = dataset['trips'].unique()

    for id in ids:
        # getting one trajectory
        trajectory = dataset[dataset['trips'] == id]
        trajectory.set_index(['trips'])

        # converting trajectory to dict
        new_dataset[id] = {}
        for col in trajectory.columns:
            new_dataset[id][col] = np.array(trajectory[col])

    return new_dataset


def get_raw_dataset(dataset_path, samples=None):
    """
    It compress the trajectories and provide compress rate and processing time.
    :param dataset_path: path of dataset with trajectories
    :param compress: compress algorithm: 'DP', 'TR', 'SP', 'DP_TR','DP_SP','TR_DP','TR_SP','SP_TR','SP_DP' (Default: 'DP')
    :param alpha: compression threshold
    """
    dataset = pd.read_csv(dataset_path, parse_dates=['time'], low_memory=False)
    dataset['time'] = dataset['time'].astype('datetime64[ns]')

    # remove invalid mmsi
    dataset = dataset[(dataset['mmsi'] < 776000000) & (dataset['mmsi'] >= 201000000)]

    if not 'trips' in dataset.columns:
        dataset = dataset.rename(columns={'trajectory': 'trips'})
    dataset = dataset.sort_values(by=['trips', "time"])
    dataset_dict = pandas_to_dict(dataset)

    if samples is not None:
        keys = list(dataset_dict.keys())[0:samples]
        dataset_dict = dict([(k, v) for k, v in dataset_dict.items() if k in keys])

    return dataset_dict