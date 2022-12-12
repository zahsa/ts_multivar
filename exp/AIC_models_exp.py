from ARmodels.ar_models import Models1
from analysis import plot_images as pli


def AIC_experiments(dataset_dict, dim_set, main_folder, measure='AIC', tr='n'):
    curr_config = {}

    print('OU')
    curr_config_ou = {}
    optimizers = ['L-BFGS-B', 'SLSQP']#, 'BFGS', 'Newton-CG', 'CG']
    for opt in optimizers:
        try:
            features = Models1(dataset=dataset_dict, features_opt='ou', optimizer=opt, dim_set=dim_set, folder=main_folder)
            curr_config_ou[f'{opt}'] = features.path
            curr_config[f'ou-{opt}'] = features.path
        except:
            print(f'ou did not run for {opt}')
    pli.info_to_plot(curr_config_ou, model='ou', measure=measure, folder=main_folder)

    print('ARIMA')
    curr_config_arima = {}
    for ar_p in [1, 2, 3]:
        for ma_p in [0, 1, 2, 3]:
            # for tr in ['c', 'n']:
            features = Models1(dataset=dataset_dict, features_opt='arima', dim_set=dim_set, ar_prm=ar_p, ma_prm=ma_p,
                               trend=tr, folder=main_folder)
            curr_config_arima[f'{ar_p}_{ma_p}_{tr}'] = features.path
            curr_config[f'arima_{ar_p}_{ma_p}_{tr}'] = features.path
    pli.info_to_plot(curr_config_arima, model='arima', measure=measure, folder=main_folder)

    print('VAR')
    curr_config_var = {}
    for ar_p in [1, 2, 3]:
        # for tr in ['n', 'c']:
        features = Models1(dataset=dataset_dict, features_opt='var', dim_set=dim_set, ar_prm=ar_p, ma_prm=0, trend=tr,
                           folder=main_folder)
        curr_config_var[f'{ar_p}_{tr}'] = features.path
        curr_config[f'var_{ar_p}_{tr}'] = features.path
    pli.info_to_plot(curr_config_var, model='var', measure=measure, folder=main_folder)

    print('MULTIARIMA')
    curr_config_multiarima = {}
    for ar_p in [1, 2, 3]:
        for ma_p in [0, 1, 2, 3]:
            # for tr in ['n','c']:
            features = Models1(dataset=dataset_dict, features_opt='multi_arima', dim_set=dim_set, ar_prm=ar_p,
                               ma_prm=ma_p, trend=tr, folder=main_folder)
            curr_config_multiarima[f'{ar_p}_{ma_p}_{tr}'] = features.path
            curr_config[f'multi_arima_{ar_p}_{ma_p}_{tr}'] = features.path

    pli.info_to_plot(curr_config_multiarima, model='multi_arima', measure=measure, folder=main_folder)

    print('VARMAX')
    curr_config_varma = {}
    for ar_p in [1, 2, 3]:
        for ma_p in [0, 1, 2, 3]:
            # for tr in ['n', 'c']:
            features = Models1(dataset=dataset_dict, features_opt='varmax', dim_set=dim_set, ar_prm=ar_p, ma_prm=ma_p,
                               trend=tr, folder=main_folder)
            curr_config_varma[f'{ar_p}_{ma_p}_{tr}'] = features.path
            curr_config[f'varma_{ar_p}_{ma_p}_{tr}'] = features.path
    pli.info_to_plot(curr_config_varma, model='varmax', measure=measure, folder=main_folder)

    pli.info_to_plot(curr_config, measure=measure, folder=main_folder)
    return curr_config


def AIC_general(dataset_dict, dim_set, main_folder, tr='n'):
    curr_config = {}

    print('OU')
    # opt = 'SLSQP'
    opt = 'L-BFGS-B'
    features = Models1(dataset=dataset_dict, features_opt='ou',  optimizer=opt, dim_set=dim_set, folder=main_folder)
    curr_config[f'ou-{opt}'] = features.path

    print('ARIMA')
    ar_p = 3
    ma_p = 0
    features = Models1(dataset=dataset_dict, features_opt='arima', dim_set=dim_set, ar_prm=ar_p, ma_prm=ma_p,
                       trend=tr, folder=main_folder)
    curr_config[f'arima_{ar_p}_{ma_p}_{tr}'] = features.path

    print('VAR')
    ar_p = 2
    features = Models1(dataset=dataset_dict, features_opt='var', dim_set=dim_set, ar_prm=ar_p, ma_prm=0, trend=tr,
                       folder=main_folder)
    curr_config[f'var_{ar_p}_{tr}'] = features.path

    # print('MULTIARIMA')
    # ar_p = 3
    # ma_p = 0
    # features = Models1(dataset=dataset_dict, features_opt='multi_arima', dim_set=dim_set, ar_prm=ar_p,
    #                    ma_prm=ma_p, trend=tr, folder=main_folder)
    # curr_config[f'multi_arima_{ar_p}_{ma_p}_{tr}'] = features.path

    # print('VARMAX')
    # ar_p = 2
    # ma_p = 0
    # features = Models1(dataset=dataset_dict, features_opt='varmax', dim_set=dim_set, ar_prm=ar_p, ma_prm=ma_p,
    #                    trend=tr, folder=main_folder)
    # curr_config[f'varma_{ar_p}_{ma_p}_{tr}'] = features.path

    return curr_config