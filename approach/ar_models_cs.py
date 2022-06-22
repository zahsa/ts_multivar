import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

from scipy.spatial.distance import pdist, squareform
import os, pickle
from joblib import Parallel, delayed
# import projection as pjt
from analysis import projection as pjt

# from approach import OU_process as ou
from sklearn.preprocessing import MinMaxScaler
import warnings
from statsmodels.tsa.stattools import adfuller


def grangers_causation_matrix(df, variables, test='ssr_chi2test', verbose=False):
    # variables = [ 'lat', 'lon', 'sog', 'cog']
    warnings.filterwarnings("ignore")

    print(f"length of time series is : {len(df)}")
    maxlag = min(10, int(len(df) / 4))
    df_pval = pd.DataFrame(np.ones((len(variables), len(variables))), columns=variables, index=variables)
    df_lag = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    # dff = pd.DataFrame.from_dict(df)

    for col in df_pval.columns:
        for row in df_pval.index:
            test_result = grangercausalitytests(df[[row, col]], maxlag=maxlag, verbose=False)
            p_values = [np.round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]
            # if verbose: print(f'Y = {row}, X = {col}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df_pval.loc[row, col] = min_p_value
            df_lag.loc[row, col] = np.argmin(p_values) + 1
    # df_pval.columns = [var + '_x' for var in variables]
    # df_pval.index = [var + '_y' for var in variables]
    return df_pval, df_lag


def dict_reorder(x):
    """
    It reorder the dict values
    :param x: the data on dict format
    :return: dict ordered
    """
    return {k: dict_reorder(v) if isinstance(v, dict) else v for k, v in sorted(x.items())}


class Models:
    """
    It reads the preprocessed DCAIS dataset in dict format.
    Next, the selected model is applied to extract the coefficients that represents the movement of the vessel trajectory.
    """

    def __init__(self, dataset, verbose=True, **args):
        """
        It receives the preprocessed DCAIS dataset in dict format.
        It applies the selected model on the trajectories and compute the euclidean distance.
        :param dataset: the dataset in dict format
        """
        self.verbose = verbose
        self.dm = None
        self.coeffs = None
        self.measures = None
        # self.num_cores = 2*(multiprocessing.cpu_count()//3)
        self.num_cores = 3
        if 'njobs' in args.keys():
            self.num_cores = args['njobs']

        self.features_opt = 'var'
        if 'features_opt' in args.keys():
            self.features_opt = args['features_opt']

        self._dim_set = ['lat', 'lon']
        if 'dim_set' in args.keys():
            self._dim_set = args['dim_set']

        self.dataset = dataset
        self._ids = list(self.dataset.keys())
        # self._ids = list(range(len(self.dataset_norm.keys())))

        _metrics_dict = self.create_data_dict()
        _metrics_dict[self.features_opt]()

        # saving features
        if 'folder' in args.keys():
            self.path = args['folder']

            if not os.path.exists(self.path):
                os.makedirs(self.path)

            df_features = pd.DataFrame(self.coeffs)
            df_features.to_csv(f'{self.path}/features_coeffs.csv')
            if self.features_opt in ['arima', 'arima_multi', 'var']:
                self.measures.to_csv(f'{self.path}/features_measures.csv')
            # pjt.plot_coeffs_traj(self.coeffs, pd.Series(np.zeros(self.coeffs.shape[0])), folder=self.path)

            pickle.dump(self.dm, open(f'{self.path}/features_distance.p', 'wb'))
            df_features = pd.DataFrame(self.dm)
            df_features.to_csv(f'{self.path}/features_distance.csv')
            self.dm_path = f'{self.path}/features_distance.p'

    def var_coefs(self):
        """
        It computes VAR model in each trajectory, and produce the distance matrix with Euclidean distance.
        """
        measure_pd = {}
        coeffs = {}
        for i in range(len(self._ids)):
            # measure_pd[self._ids[i]] = {}
            # coeffs[self._ids[i]] = {}
            measure_pd[i] = {}
            coeffs[i] = {}
        # Parallel(n_jobs=self.num_cores, require='sharedmem')(delayed(self._var_func)(i, measure_pd, coeffs) for i in list(range(len(self._ids))))

        for i in list(range(len(self._ids))):
        # for i in list(range(2)):
            self._var_func(i, measure_pd, coeffs)

        measure_pd = pd.DataFrame(measure_pd).T

        col_names_all = ['AIC', 'BIC', 'fpe', 'HQIC']

        col_names = ['mmsi']
        # for dim in range(len(self._dim_set)):
        #     col_names = col_names + col_names_all

        col_names = col_names + col_names_all

        measure_pd.columns = col_names
        self.measures = measure_pd

        coeffs = dict_reorder(coeffs)
        self.coeffs = np.array([coeffs[item] for item in coeffs.keys()])
        self.coeffs[np.isnan(self.coeffs)] = 0

        scaler = MinMaxScaler().fit(self.coeffs)
        self.coeffs = scaler.transform(self.coeffs)

        # self.dm = squareform(pdist(self.coeffs))
        # self.dm = self.dm / self.dm.max()

    def _var_func(self, i, measure_pd, coeffs):
        """
        It computes VAR model for one trajectory.
        """
        coeffs_i = np.array([])

        # import ipdb;ipdb.set_trace()
        measure_list = [self.dataset[self._ids[i]]['mmsi'][0]]
        if self.verbose:
            print(f"VAR Computing {i} of {len(self._ids)}")

        dff = pd.DataFrame.from_dict(self.dataset[self._ids[i]])

        df_features = dff.loc[:, ['lat_norm', 'lon_norm']] #, 'sog', 'cog']]

        print(f'df_feat for feat{i}')

        model_var = VAR(df_features)
        res = model_var.fit(2)

        rp = res.params
        coef1 = rp.loc[:, "lon_norm"].values
        coeffs_i = np.hstack((coeffs_i, coef1))

        coef2 = rp.loc[:, "lat_norm"].values
        coeffs_i = np.hstack((coeffs_i, coef2))

        print('coef shapes', coef1.shape, coef2.shape)

        measure_list = measure_list + [res.aic, res.bic, res.fpe, res.hqic]

        measure_pd[i] = measure_list
        coeffs[i] = coeffs_i

    def create_data_dict(self):
        """
        Dictionary of models options.
        """
        return {
            'var': self.var_coefs}

