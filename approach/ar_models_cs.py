import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

from scipy.spatial.distance import pdist, squareform
import os, pickle
from joblib import Parallel, delayed
#import projection as pjt
from analysis import projection as pjt

from approach import OU_process as ou
from sklearn.preprocessing import MinMaxScaler
import warnings
from statsmodels.tsa.stattools import adfuller

def grangers_causation_matrix(df, variables, test='ssr_chi2test', verbose=False):
   
    # variables = [ 'lat', 'lon', 'sog', 'cog']
    warnings.filterwarnings("ignore")

    print(f"length of time series is : {len(df)}")
    maxlag = min(10,int(len(df)/4))
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
            df_lag.loc[row, col] = np.argmin(p_values)+1
    # df_pval.columns = [var + '_x' for var in variables]
    # df_pval.index = [var + '_y' for var in variables]
    return df_pval,df_lag

def dict_reorder(x):
    """
    It reorder the dict values
    :param x: the data on dict format
    :return: dict ordered
    """
    return {k: dict_reorder(v) if isinstance(v, dict) else v for k, v in sorted(x.items())}


def avg_std_dict_data(x, dim_set):
    """
    It computes the average and the standard deviation of a dict dataset considering a set of atributes
    :param x: the dataset in dict format
    :param dim_set: a list of the attributes to be computed
    :return: average and standard deviation
    """
    avg = {}
    std = {}
    maxv = {}
    for dim in dim_set:
        aux = np.concatenate([x[k].get(dim) for k in x])
        avg[dim] = aux.mean()
        std[dim] = aux.std()
        maxv[dim] = aux.max()
    return avg, std, maxv


def normalize(x, dim_set, verbose=True, znorm=True, centralize=False, norm_geo=True):
    """
    Computes Z-normalization or centralization of a dict dataset for a set of attributes
    :param x: dict dataset
    :param dim_set: set of attributes
    :param verbose: if True, print comments
    :param znorm: if True, it computes the z-normalization
    :param centralize: it True, it computes the centralization
    :return: normalized dict dataset
    """
    if verbose:
        print(f"Normalizing dataset")
    avg, std, maxv = avg_std_dict_data(x, dim_set)

    print(f'x length {len(x.keys())} before prune')

    ids = list(x.keys())
    for id_a in range(len(ids)):
        # normalize features

        if znorm:
            for dim in dim_set:
                x[ids[id_a]][dim] = (x[ids[id_a]][dim]-avg[dim]) / std[dim]
        elif centralize:
            for dim in dim_set:
                x[ids[id_a]][dim] = x[ids[id_a]][dim]-avg[dim]
        elif norm_geo:
            for dim in dim_set:
                if (dim == 'lat') or (dim == 'LAT'):
                    x[ids[id_a]][dim] = x[ids[id_a]][dim]/90
                elif (dim == 'lon') or (dim == 'LON'):
                    x[ids[id_a]][dim] = x[ids[id_a]][dim]/180
                else:
                    # x[ids[id_a]][dim] = x[ids[id_a]][dim]/maxv[dim]
                    x[ids[id_a]][dim] = (x[ids[id_a]][dim]-avg[dim]) / std[dim]

    x2 = x.copy()

    # all the columns with constant values need to be removed

    for key in x:
        # normalize features
        for dim in dim_set:
            if ((dim == 'lat') or (dim == 'LAT')):
                xlat = x2[key][dim]
                if np.var(xlat) < 1e-3:
                    print(f'remove constant series with key {key}')
                    del x2[key]
                elif ((dim == 'lon') or (dim == 'LON')):

                    xlon = x2[key][dim]
                    if np.var(xlon) < 1e-3:
                        print(f'remove constant series with key {key}')
                        del x2[key]


    print(f'x length {len(x2.keys())} after prune')

   
    return x2


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

        self._znorm = False
        if 'znorm' in args.keys():
            self._znorm = args['znorm']

        self._centralize = False
        if 'centralize' in args.keys():
            self._centralize = args['centralize']

        self._normalizationGeo = True
        if 'norm_geo' in args.keys():
            self._normalizationGeo = args['norm_geo']

        self.dataset_norm = dataset
        if self._znorm or self._centralize or self._normalizationGeo:
            self.dataset_norm = normalize(self.dataset_norm, self._dim_set, verbose=verbose, znorm=self._znorm, centralize=self._centralize, norm_geo=self._normalizationGeo)
        self._ids = list(self.dataset_norm.keys())
        # self._ids = list(range(len(self.dataset_norm.keys())))


        # methods parameters
        self.ar_prm = 1
        if 'ar_prm' in args.keys():
            self.ar_prm = args['ar_prm']

        self.ma_prm = 1
        if 'ma_prm' in args.keys():
            self.ma_prm = args['ma_prm']

        self.i_prm = 0
        if 'i_prm' in args.keys():
            self.i_prm = args['i_prm']

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


        #for i in list(range(len(self._ids))):
        for i in list(range(2)):
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


   
   
    def stationary_test(self, signal):
        if adfuller(signal)[1] >= 0.05:
            print('signal is non-stationary')
            return(signal.diff().dropna())
        else:
            return(signal)


    def _var_func(self, i, measure_pd, coeffs):
        """
        It computes VAR model for one trajectory.
        """
        coeffs_i = np.array([])


        measure_list = [self.dataset_norm[self._ids[i]]['mmsi'][0]]
        if self.verbose:
            print(f"VAR Computing {i} of {len(self._ids)}")


        dff = pd.DataFrame.from_dict(self.dataset_norm[self._ids[i]])
       
###################################################
        df_columns = [ 'lat', 'lon', 'sog', 'cog'];dflist = []
        for dc in df_columns:
            series = dff.loc[:,dc]
            dframe = pd.DataFrame(series)
            statres = adfuller(dframe.values)
            if statres[1]>=0.05:
                print(f'{dc} is not stationary')
                stat_df = dframe.diff().dropna()
            else:
                print(f'{dc} is stationary')
                stat_df = dframe
            dflist.append(stat_df)
        min_len = np.min([len(d) for d in dflist])
        dflist2 = [dl[0:min_len] for dl in dflist]
        [dl.reset_index(drop=True, inplace=True) for dl in dflist2]
        dff_stat = pd.concat(dflist2,axis=1)


        df_features = dff_stat
        model_var = VAR(df_features)
        model_var.select_order(15)
        res = model_var.fit(maxlags=15, ic='aic')
        # irf = results.irf(10)

        print('var res')

        rp = res.params
        for dc in df_columns:
            coef = rp.loc[:, dc].values
            coeffs_i = np.hstack((coeffs_i,coef))

        
        measure_list = measure_list + [res.aic, res.bic, res.fpe, res.hqic]

        measure_pd[i] = measure_list
        coeffs[i] = coeffs_i
    def create_data_dict(self):
        """
        Dictionary of models options.
        """
        return {
                'var': self.var_coefs}


