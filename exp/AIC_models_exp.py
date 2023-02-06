from ARmodels.ar_models import Models1
from analysis import plot_images as pli


def AIC_experiments(dataset_dict, dim_set, main_folder, measure='AIC', tr='n'):
    curr_config = {}

    print('OU')
    curr_config_ou = {}
    # optimizers = ['L-BFGS-B', 'SLSQP', 'BFGS', 'Newton-CG', 'CG']
    optimizers = ['SLSQP']
    for opt in optimizers:
        try:
            features = Models1(dataset=dataset_dict, features_opt='ou', optimizer=opt, dim_set=dim_set, folder=main_folder)
            curr_config_ou[f'{opt}'] = features.path
            curr_config[f'OU'] = features.path
        except:
            print(f'ou did not run for {opt}')
    pli.info_to_plot(curr_config_ou, model='ou', measure=measure, folder=main_folder)

    print('ARIMA')
    curr_config_arima = {}
    for ar_p in [1, 2, 3, 4]:
        for ma_p in [0, 1, 2, 3]:
            # for tr in ['c', 'n']:
            features = Models1(dataset=dataset_dict, features_opt='arima', dim_set=dim_set, ar_prm=ar_p, ma_prm=ma_p,
                               trend=tr, folder=main_folder)
            curr_config_arima[f'{ar_p}_{ma_p}'] = features.path
            curr_config[f'ARMA({ar_p},{ma_p})'] = features.path
    pli.info_to_plot(curr_config_arima, model='arima', measure=measure, folder=main_folder)

    print('VAR')
    curr_config_var = {}
    for ar_p in [1, 2, 3, 4]:
        # for tr in ['n', 'c']:
        features = Models1(dataset=dataset_dict, features_opt='var', dim_set=dim_set, ar_prm=ar_p, ma_prm=0, trend=tr,
                           folder=main_folder)
        curr_config_var[f'{ar_p}_{tr}'] = features.path
        curr_config[f'VAR({ar_p}) CML'] = features.path
    pli.info_to_plot(curr_config_var, model='var', measure=measure, folder=main_folder)

    # print('MULTIARIMA')
    # curr_config_multiarima = {}
    # for ar_p in [1, 2, 3]:
    #     for ma_p in [0, 1, 2, 3]:
    #         # for tr in ['n','c']:
    #         features = Models1(dataset=dataset_dict, features_opt='multi_arima', dim_set=dim_set, ar_prm=ar_p,
    #                            ma_prm=ma_p, trend=tr, folder=main_folder)
    #         curr_config_multiarima[f'{ar_p}_{ma_p}_{tr}'] = features.path
    #         curr_config[f'multi_arima_{ar_p}_{ma_p}_{tr}'] = features.path
    #
    # pli.info_to_plot(curr_config_multiarima, model='multi_arima', measure=measure, folder=main_folder)

    print('VARMAX')
    curr_config_varma = {}
    for ar_p in [1, 2, 3, 4]:
        # for ma_p in [0, 1, 2, 3]:
        # for tr in ['n', 'c']:
        ma_p = 0
        features = Models1(dataset=dataset_dict, features_opt='varmax', dim_set=dim_set, ar_prm=ar_p, ma_prm=ma_p,
                           trend=tr, folder=main_folder)
        curr_config_varma[f'{ar_p}_{ma_p}_{tr}'] = features.path
        curr_config[f'VAR({ar_p}) EML'] = features.path
    pli.info_to_plot(curr_config_varma, model='varmax', measure=measure, folder=main_folder)

    pli.info_to_plot(curr_config, measure=measure, folder=main_folder)
    return curr_config


def AIC_general(dataset_dict, dim_set, main_folder, tr='n', ou_opt='L-BFGS-B',
                arima_ar_p=3, arima_ma_p=0, var_ar_p=2, varmax_ar_p=2, varmax_ma_p=0, verbose=False):
    curr_config = {}

    if verbose:
        print('OU')
    features = Models1(dataset=dataset_dict, features_opt='ou',  optimizer=ou_opt, dim_set=dim_set, folder=main_folder)
    curr_config[f'ou-{ou_opt}'] = features.path

    if verbose:
        print('ARIMA')
    features = Models1(dataset=dataset_dict, features_opt='arima', dim_set=dim_set, ar_prm=arima_ar_p, ma_prm=arima_ma_p,
                       trend=tr, folder=main_folder)
    curr_config[f'arima_{arima_ar_p}_{arima_ma_p}_{tr}'] = features.path

    if verbose:
        print('VAR')
    features = Models1(dataset=dataset_dict, features_opt='var', dim_set=dim_set, ar_prm=var_ar_p, ma_prm=0, trend=tr,
                       folder=main_folder)
    curr_config[f'var_{var_ar_p}_{tr}'] = features.path

    # if verbose:
    #   print('VARMAX')
    # features = Models1(dataset=dataset_dict, features_opt='varmax', dim_set=dim_set, ar_prm=varmax_ar_p, ma_prm=varmax_ma_p,
    #                    trend=tr, folder=main_folder)
    # curr_config[f'varma_{varmax_ar_p}_{varmax_ma_p}_{tr}'] = features.path

    return curr_config