import os, random
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from datetime import timedelta
import warnings
import ssl
from matplotlib import pyplot as plt
from haversine import haversine

ssl._create_default_https_context = ssl._create_unverified_context


def normalize(x, dim_set, verbose=True):
    """
    Divides my maximum if lat and lon and computes Z-normalization for remaining attributes
    :param x: pandas dataset
    :param dim_set: set of attributes
    :param verbose: if True, print comments
    :return: normalized dataset
    """
    if verbose:
        print(f"Normalizing dataset")
    avg = x[dim_set].mean(axis=0)
    std = x[dim_set].std(axis=0)
    # maxv = x[dim_set].max(axis=0)
    x['lat_norm'] = x['lat'] / 90
    x['lon_norm'] = x['lon'] / 180

    for dim in dim_set:
        if (dim != 'lat') and (dim != 'lon'):
            x[f'{dim}_norm'] = (x[dim] - avg[dim]) / std[dim]

    return x


def missing_values_treatment(x):
    """
    It removes observations with invalid SOG and COG.
    :param x: the dataset
    :return: the dataset without invalid samples
    """
    # missing and invalid values in sog and cog are removed
    if 'sog' in x.columns:
        x.loc[x['sog'] == 'None', 'sog'] = -1
        x.loc[x['sog'] > 60, 'sog'] = -1
        x.sog = x.sog.astype('float64')
        x = x.drop(x[x['sog'] == -1].index)
    if 'cog' in x.columns:
        x.loc[x['cog'] < 0, 'cog'] = -1
        x.loc[x['cog'] == 'None', 'cog'] = -1
        x.cog = x.cog.astype('float64')
        x = x.drop(x[x['cog'] == -1].index)

    return x


def removing_invalid_samples(x, min_obs=None, subset=None):
    """
    It round the values to 4 decimals, removes duplicates, and removes samples with few observations.
    :param x: the dataset
    :return: the dataset with country attribute
    """
    # round values to 4 decimals (10 meters)
    # x.lat = x.lat.round(4)
    # x.lon = x.lon.round(4)

    # remove duplicate entries
    x = x.drop_duplicates(subset=subset, keep='first')

    if min_obs is not None:
        # remove mmsi with less than min observations
        obs_per_mmsi = x.groupby(x['mmsi'], as_index=False).size()
        ids_to_keep = obs_per_mmsi['mmsi'][obs_per_mmsi['size'] >= min_obs]
        x = x[x['mmsi'].isin(ids_to_keep)]

    return x


def include_country(x):
    """
    It includes the country based on the MMSI
    :param x: the dataset
    :return: the dataset with country attribute
    """
    # include flags
    MMIS_digits_path = './data/MaritimeIdentificationDigits.csv'
    if os.path.exists(MMIS_digits_path):
        MID = pd.read_csv(MMIS_digits_path)
        flag_col = x['mmsi']
        flag_col = flag_col // 1000000
        flag_col = flag_col.replace(MID.set_index('Digit')['Allocated to'])
        x = x.assign(flag=pd.Series(flag_col, index=x.index))
    else:
        warnings.warn(f"File {MMIS_digits_path} was not found.")
    return x


### Reading and filtering dataset ###
def date_range(start_date, end_date):
    """
    It provides ranges of date period to conduct the loop
    :param start_date: initial date period to get the dataset
    :param end_date: final date period to get the dataset
    :return: iterative data
    """
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def create_dataset_noaa(path, time_period, vt=None):
    """
    It reads the noaa dataset and produce a csv file with the vessels information of a specific type.
    Such vessel type provide the most trips information.
    :param time_period: initial and final date period to get the dataset
    :param vt: vessel type
    :return: path where the csv file was saved
    """
    columns_read = ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG', 'Heading', 'VesselType', 'Length', 'Width']
    dataset = pd.DataFrame()
    mmsis = np.array([])
    for curr_date in date_range(time_period[0], time_period[1]+timedelta(days=1)):
        print(f'\treading day {curr_date}')
        url = urlopen(f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2020/AIS_2020_{curr_date.month:02d}_{curr_date.day:02d}.zip")
        file_name = f'AIS_2020_{curr_date.month:02d}_{curr_date.day:02d}.csv'
        zipfile = ZipFile(BytesIO(url.read()))
        chunk = pd.read_csv(zipfile.open(file_name), usecols=columns_read)
        if vt is not None:
            # chunk2 = chunk[chunk['VesselType'] == vt]
            chunk2 = chunk[chunk['VesselType'].isin(vt)]
            mmsis = np.concatenate((mmsis, chunk2['MMSI'].unique()))
            mmsis = np.unique(mmsis)
            chunk = chunk[chunk['MMSI'].isin(mmsis)]
        dataset = pd.concat([dataset, chunk], ignore_index=True)
        zipfile.close()

    dataset['VesselType'] = dataset['VesselType'].fillna(-1)
    dataset['VesselType'] = dataset['VesselType'].astype(int)

    dataset = dataset[['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG', 'VesselType']]
    dataset.columns = ['mmsi', 'time', 'lat', 'lon', 'sog', 'cog', 'vessel_type']

    dataset.to_csv(path, index=False)

### Class to produce the preprocessed dataset ###
class Trajectories:
    """
    It reads the DCAIS dataset and produce a csv file with the preprocessed vessels information.
    The dataset corresponds to a specific vessel type for a particular period of time.
    It reads, clean and aggregate information of the vessels.
    """
    def __init__(self, n_samples=None, vessel_type=None, time_period=None, min_obs=100, **args):
        """
        It reads the noaa dataset and produce a csv file with the vessels information of a specific type.
        Such vessel type provide the most trips information.
        :param n_samples: number of MMSI to be processed, none if you request all MMSIs (Default: None)
        :param vessel_typet: vessel type
        :param time_period: period of time to read the dataset
        :param min_obs: minimum number of observations
        """
        self._nsamples = n_samples
        self._vt = vessel_type
        self._vessel_types = None
        self._columns_set = ['lat', 'lon', 'cog', 'sog']
        # it just considers trajectories with more than such number of observations
        self.min_obs = min_obs
        if self.min_obs < 2:
            self.min_obs = 2

        self.region = None
        if 'region' in args.keys():
            self.region = args['region']

        self.threshold = 1e-5
        if 'threshold' in args.keys():
            self.threshold = args['threshold']

        if time_period is None:
            time_period = (datetime(2020, 4, 19), datetime(2020, 4, 25))

        if not os.path.exists('./data/preprocessed/'):
            os.makedirs('./data/preprocessed/')

        day_name = f'{time_period[0].day:02d}-{time_period[0].month:02d}_to_{time_period[1].day:02d}-{time_period[1].month:02d}'
        self.dataset_path = f"./data/preprocessed/DCAIS_vessels_{self._vt}_{day_name}_time_period.csv"
        self.cleaned_path = f"./data/preprocessed/DCAIS_vessels_{self._vt}_{day_name}_clean.csv"
        self.segmentation_path = f"./data/preprocessed/DCAIS_vessels_{self._vt}_{day_name}_trips.csv"
        self.preprocessed_path = f"./data/preprocessed/DCAIS_vessels_{self._vt}_{self._nsamples}-mmsi_{day_name}_trips.csv"
        # self.preprocessed_path = f"./data/preprocessed/DCAIS_vessels_{self._vt}_{self._nsamples}-mmsi_{day_name}_trips_prune.csv"

        if self.region is not None:
            self.preprocessed_path = f"./data/preprocessed/DCAIS_vessels_{self._vt}_{self._nsamples}-mmsi_region_{self.region}_{day_name}_trips.csv"
            # self.preprocessed_path = f"./data/preprocessed/DCAIS_vessels_{self._vt}_{self._nsamples}-mmsi_region_{self.region}_{day_name}_trips_prune.csv"

        if not os.path.exists(self.dataset_path):
            create_dataset_noaa(self.dataset_path, vt=self._vt, time_period=time_period)
            print(f'Preprocessed data save at: {self.dataset_path}')
        else:
            print('path1 exists')

        if not os.path.exists(self.cleaned_path):
            self.cleaning()
            print(f'Clean data save at: {self.cleaned_path}')
        else:
            print('path2 exists')

        if not os.path.exists(self.segmentation_path):
            self.trips_segmentation()
            print(f'Clean data save at: {self.segmentation_path}')
        else:
            print('path3 exists')

        if not os.path.exists(self.preprocessed_path):
            self.mmsi_trips_prune()
            print(f'Preprocessed trips data save at: {self.preprocessed_path}')
        else:
            print('path4 exists')

    def cleaning(self):
        """
        It cleans the dataset, removing invalid samples and including country information.
        """
        # reading dataset of a time period
        dataset = pd.read_csv(self.dataset_path, parse_dates=['time'])
        dataset['time'] = dataset['time'].astype('datetime64[ns]')
        dataset = dataset.sort_values(['time'])

        # removing invalid data
        dataset = removing_invalid_samples(dataset, min_obs=self.min_obs, subset=['mmsi', 'time'])
        # missing values are replaced to -1 and removed
        dataset = missing_values_treatment(dataset)
        # including country information
        dataset = include_country(dataset)

        dataset.to_csv(self.cleaned_path, index=False)

    def plot_df(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
        plt.figure(figsize=(16, 5), dpi=dpi)
        plt.plot(x, y, color='tab:red')
        plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
        plt.show()

    def trips_segmentation(self):
        # reading dataset of a time period
        dataset = pd.read_csv(self.cleaned_path, parse_dates=['time'])
        dataset['time'] = dataset['time'].astype('datetime64[ns]')
        dataset = dataset.sort_values(by=['mmsi', "time"])
        new_dataset = pd.DataFrame()

        ids = dataset['mmsi'].unique()
        trips = 0
        # create trajectories
        count_mmsi = 0
        for id in ids:
            print(f'\t Segmenting trajectory {count_mmsi} of {len(ids)}')
            trajectory = dataset[dataset['mmsi'] == id]

            #segment by time
            duration_step = trajectory['time'].diff()
            idx = duration_step[duration_step.apply(lambda x: x.days) > 1].index.to_list()

            #segment by distance
            def pandas_dist(x, y):
                return haversine(x, y)

            trajectory['point'] = list(zip(trajectory['lat'], trajectory['lon']))
            trajectory['point1'] = trajectory['point'].shift(1)
            trajectory['point1'] = trajectory['point1'].combine_first(trajectory.point)
            distance_step = trajectory.apply(lambda x: pandas_dist(x['point'], x['point1']), axis=1)
            idx2 = distance_step[distance_step > 10].index.to_list()
            idx = list(set(idx + idx2))

            idx.append(duration_step.index[-1])
            idx.sort()
            init = 0
            for k in idx:
                trajectory.loc[init:k,'trips'] = trips
                trips = trips + 1
                init = k
            trips = trips + 1

            #remove unecessary columns
            trajectory.drop(['point', 'point1'], axis=1, inplace=True)
            # add trajectory
            new_dataset = pd.concat([new_dataset, trajectory], axis=0, ignore_index=True)
            count_mmsi = count_mmsi + 1

        #remove trips with less min obs
        count_trips = new_dataset.groupby('trips').count()
        idx = count_trips[count_trips['mmsi'] < self.min_obs].index
        new_dataset = new_dataset[new_dataset['trips'].isin(idx) == False]

        #save file
        new_dataset.to_csv(self.segmentation_path, index=False)


    def mmsi_trips_prune(self):
        """
        It reads the DCAIS dataset, select MMSI randomly if a number of samples is defined.
        It process the trajectories of each MMSI (pandas format).
        Save the dataset in a csv file.
        """
        # reading dataset of a time period
        dataset = pd.read_csv(self.segmentation_path, parse_dates=['time'])
        dataset['time'] = dataset['time'].astype('datetime64[ns]')
        dataset = dataset.sort_values(by=['mmsi', "time"])

        # select mmsi randomly
        ids = dataset['trips'].unique()

        dataset = normalize(dataset, ['lat', 'lon'])

        new_dataset = pd.DataFrame()
        # create trajectories
        count_trips = 0
        for id in ids:
            print(f'\t Cleaning trajectory {count_trips} of {len(ids)}')
            trajectory = dataset[dataset['trips'] == id] #z: collecting all rows with the same id, gives us a trajectory of one vessel

            # selecting the region
            isin_region = True
            if self.region is not None:
                if (trajectory['lat'].between(self.region[0], self.region[1]).sum() == 0) | (
                        trajectory['lon'].between(self.region[2], self.region[3]).sum() == 0):
                    isin_region = False

            # if is inside the selected region and contains enough observations
            if (trajectory.shape[0] >= self.min_obs) and isin_region:
                # remove trajectories with constant values
                if np.var(trajectory.lat_norm) > self.threshold and np.var(trajectory.lon_norm) > self.threshold:
                    # add trajectory
                    new_dataset = pd.concat([new_dataset, trajectory], axis=0, ignore_index=True)
                    count_trips = count_trips + 1
                else:
                    print(f'\t\ttrajectory {id} is removed')
            # count_mmsi = count_mmsi + 1 #z if one trajectory is removed then the whole information for one vessel is deleted
         
        self._nsamples = count_trips
        print(f'{count_trips} remaining')

        new_dataset.to_csv(self.preprocessed_path, index=False)

    def pandas_to_dict(self):
        """
        It converts the csv dataset into dict format.
        :return: dataset in a dict format.
        """
        # reading cleaned data

        dataset = pd.read_csv(self.preprocessed_path, parse_dates=['time'])


        dataset['time'] = dataset['time'].astype('datetime64[ns]')
        dataset = dataset.sort_values(by=['trips', "time"])

        new_dataset = {}
        ids = dataset['trips'].unique()
        self._nsamples = len(ids)

        for id in ids:  # is is corresponding to vessels ids?
            # getting one trajectory
            trajectory = dataset[dataset['trips'] == id]  ## what is trajectory column??
          
            trajectory.set_index(['trips'])

            # converting trajectory to dict
            new_dataset[id] = {}
            for col in trajectory.columns:
                new_dataset[id][col] = np.array(trajectory[col]) # trajectory has columns of features values ????

        return new_dataset

