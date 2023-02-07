import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib

matplotlib.use('Agg')


def boxplot_model(data, measure='AIC', model='arima', folder='./images/'):
    dim = 'Latitude and Longitude'

    x = pd.DataFrame()
    for i in data.keys():
        ids_0 = data[i][data[i][measure] == 0].index
        data[i] = data[i].drop(ids_0)
        data[i][measure] = data[i][measure][data[i][measure] < 1e4]
        if f'{measure}.1' in data[i].keys():
            data[i][f'{measure}.1'] = data[i][f'{measure}.1'][data[i][f'{measure}.1'] < 1e4]
            x = pd.concat([x, (data[i][measure] + data[i][f'{measure}.1']) / 2], axis=1)
        else:
            x = pd.concat([x, data[i][measure]], axis=1)

    x.columns = data.keys()
    x = x.replace([np.inf], np.nan)
    x = x.replace([np.nan], x.max().max())
    x = x.iloc[0:100, :]
    print(x.shape)

    # Boxplot
    fig = go.Figure()
    for i in x.columns:
        fig.add_trace(go.Box(y=x[i],
                                name=i))
    # fig.update_traces(box_visible=True, meanline_visible=True, showlegend=False)
    fig.update_traces(showlegend=False)
    fig.update_layout(title_text=model, width=1000, height=700)
    fig.update_yaxes(tickfont=dict(size=16))
    # fig.show()
    fig.write_image(f'{folder}/features_{measure}_{dim}_{model}.png', scale=1)

    table = pd.concat([x.mean(), x.std(), x.max(), x.min(), x.quantile(0.25), x.median(), x.quantile(0.75)], axis=1)
    table.to_csv(f'{folder}/{measure}_{dim}_stats.csv')
    print('AIC', end=" ")
    for i in range(table.shape[0]):
        print(f'& {round(table.iloc[i,4], 2)}',end=" ")
    print('//')

    # BAR CHART MEDIAN
    fig = go.Figure(go.Bar(name='median', y=table.index, x=table[4], text=round(table[4], 2)))
    # Change the bar mode
    fig.update_traces(orientation='h', textposition='inside', textfont_size=16)
    fig.update_layout(width=1300, height=1000, barmode='relative', uniformtext_mode='hide')
    fig.update_yaxes(tickfont=dict(size=16))
    # fig.show()
    fig.write_image(f'{folder}/features_median_{model}.png', scale=1)

    # BAR CHART ALL
    fig = go.Figure(data=[
        go.Bar(name='max', y=table.index, x=table[2], text=round(table[2], 2)),
        go.Bar(name='median', y=table.index, x=table[4], text=round(table[4], 2)),
        go.Bar(name='mean', y=table.index, x=table[0], text=round(table[0], 2))
    ])
    # Change the bar mode
    fig.update_traces(orientation='h', textposition='inside', textfont_size=16)
    fig.update_layout(width=1300, height=1000, barmode='relative', uniformtext_mode='hide', uniformtext_minsize=10)
    fig.update_yaxes(tickfont=dict(size=16))
    # fig.show()
    fig.write_image(f'{folder}/features_median_max_mean_{model}.png', scale=1)


def boxplot_all(data, measure='AIC', folder='./images/'):
    dim = 'Latitude and Longitude'

    x = pd.DataFrame()
    col = {}

    for i in data.keys():
        ids_0 = data[i][data[i][measure] == 0].index
        data[i] = data[i].drop(ids_0)
        data[i][measure] = data[i][measure][data[i][measure] < 1e4]
        if f'{measure}.1' in data[i].keys():
            data[i][f'{measure}.1'] = data[i][f'{measure}.1'][data[i][f'{measure}.1'] < 1e4]
            x = pd.concat([x, (data[i][measure] + data[i][f'{measure}.1']) / 2], axis=1)
            if 'OU' in i:
                col[i] = 'orange'
            elif 'multi_arima' in i:
                col[i] = 'blue'
            elif 'ARMA' in i:
                col[i] = 'green'
        else:
            x = pd.concat([x, data[i][measure]], axis=1)
            if 'EML' in i:
                col[i] = 'black'
            elif 'CML' in i:
                col[i] = 'red'

    x.columns = data.keys()
    x = x.replace([np.inf], np.nan)
    x = x.replace([np.nan], x.max().max())
    x = x.iloc[0:100, :]
    print(x.shape)
    # Plot

    fig = go.Figure()
    for i in x.columns:
        fig.add_trace(go.Box(y=x[i],
                                name=i,
                                line_color=col[i]))
    fig.update_layout(width=1300, height=1000)
    # fig.show()
    fig.write_image(f'{folder}/features_{measure}_{dim}_all.png', scale=3)

    table = pd.concat([x.mean(), x.std(), x.max(), x.min(), x.quantile(0.25), x.median(), x.quantile(0.75)], axis=1)
    table.columns = ['mean', 'std', 'max', 'min', 'quart25', 'median', 'quart75']
    table.to_csv(f'{folder}/{measure}_{dim}_stats.csv')

    # BAR CHART MEDIAN
    fig = go.Figure(go.Bar(name='median', y=table.index, x=table['median'], text=round(table['median'], 2)))
    # Change the bar mode
    fig.update_traces(orientation='h', textposition='inside', textfont_size=20)
    fig.update_layout(width=1300, height=1000, barmode='relative', uniformtext_mode='hide', font=dict(size=20))
    # fig.show()
    fig.write_image(f'{folder}/features_median_all.png', scale=1)

    # BAR CHART ALL
    fig = go.Figure(data=[
        go.Bar(name='max', y=table.index, x=table['max'], text=round(table['max'], 2)),
        go.Bar(name='median', y=table.index, x=table['median'], text=round(table['median'], 2)),
        go.Bar(name='mean', y=table.index, x=table['mean'], text=round(table['mean'], 2))
    ])
    # Change the bar mode
    fig.update_traces(orientation='h', textposition='inside', textfont_size=16)
    fig.update_layout(barmode='relative', uniformtext_minsize=10)
    # fig.show()
    fig.update_layout(width=1300, height=1000)
    fig.write_image(f'{folder}/features_median_max_mean_all.png', scale=1)


def plot_time(data, model='all', folder='./images/'):
    x = pd.DataFrame()
    for i in data.keys():
        time_pd = data[i].T
        time_rate = time_pd['time']/time_pd['lenght']
        curr_c = pd.DataFrame([time_rate.mean()], columns=[i])
        x = pd.concat([x, curr_c], axis=1)
    x.index = ['Time Rate per trajectory (s)']
    fig = px.bar(x.T, y='Time Rate per trajectory (s)', title=model, text_auto='.4f')
    fig.update_traces(textfont_size=20, textangle=0, textposition="outside", cliponaxis=False)
    fig.update_yaxes(tickfont=dict(size=20))
    fig.update_layout(width=1300, height=1000)
    # fig.show()
    fig.write_image(f'{folder}/time_rate_{model}.png', scale=3)


def info_to_plot(data, model=None, measure='AIC', folder='./images/'):
    info_data = {}
    info_data_time = {}

    for i in data.keys():
        file_path = f'{data[i]}/features_measures.csv'
        file_path_time = f'{data[i]}/features_processing_time.csv'

        info_data[i] = pd.read_csv(file_path)
        info_data_time[i] = pd.read_csv(file_path_time, index_col=0)

    if model is None:
        boxplot_all(info_data, measure=measure, folder=folder)
        plot_time(info_data_time, model='all', folder=folder)
    else:
        boxplot_model(info_data, model=model, measure=measure, folder=folder)
        plot_time(info_data_time, model=model, folder=folder)


def plot_all(data, folder='./images/', verbose=False):
    info_data = {}
    info_data_time = {}

    for c in data[37].keys():
    # for c in ['ou-L-BFGS-B', 'ou-SLSQP', 'arima_1_0_n']:
        if verbose:
            print(c)
        config = pd.DataFrame()
        config_time = pd.DataFrame()
        for type_v in data.keys():
            if c in data[type_v].keys():
                file_path = f'{data[type_v][c]}/features_measures.csv'
                file_path_time = f'{data[type_v][c]}/features_processing_time.csv'
                features = pd.read_csv(file_path)
                info_time = pd.read_csv(file_path_time, index_col=0)
                config = pd.concat([config, features], axis=0)
                config_time = pd.concat([config_time, info_time.T], axis=0)

        info_data[c] = config['AIC']
        info_data_time[c] = config_time['time']#/config_time['lenght']

    #reset index
    info_data_d = {}
    for k, v in info_data.items():
        aux = pd.Series(v)
        aux = aux.reset_index(drop=True)
        info_data_d[k] = aux

    info_data_time_d = {}
    for k, v in info_data_time.items():
        aux = pd.Series(v)
        aux = aux.reset_index(drop=True)
        info_data_time_d[k] = aux

    config = pd.DataFrame.from_dict(info_data_d)
    config.fillna(config.max().max())
    config[config > 1e5] = 1e5
    config[config < -5e5] = -5e5
    config_time = pd.DataFrame.from_dict(info_data_time_d)

    col = {}
    for i in config.keys():
        if 'OU' in i:
            col[i] = 'orange'
        elif 'multi_arima' in i:
            col[i] = 'blue'
        elif 'ARMA' in i:
            col[i] = 'green'
        else:
            if 'EML' in i:
                col[i] = 'black'
            elif 'CML' in i:
                col[i] = 'red'

    # box plot
    order = config.median(axis=0).sort_values()
    fig = go.Figure()
    for i in order.index:
        fig.add_trace(go.Box(y=config[i],
                             name=i,
                             line_color=col[i]))
    fig.update_traces(showlegend=False)
    fig.update_layout(width=1300, height=1000, font=dict(size=20))
    fig.update_yaxes(tickfont=dict(size=20), title='AIC values')
    fig.update_xaxes(tickfont=dict(size=20), title='Models Configuration')
    # fig.show()
    fig.write_image(f'{folder}/features_AIC_all.png', scale=1)

    # time plot

    x = config_time.mean()
    x = x.sort_values()
    fig = px.bar(x, text_auto='.2f')
    fig.update_yaxes(tickfont=dict(size=20), title='Average of the processing time (s)')
    fig.update_xaxes(tickfont=dict(size=20), title='Models Configuration')
    fig.update_layout(width=1300, height=1000, showlegend=False, uniformtext_minsize=16, font=dict(size=20))
    fig.update_traces(textfont_size=20, textangle=90, cliponaxis=False)
    fig['data'][0].width = 1.2
    # fig.show()
    fig.write_image(f'{folder}/time_rate_all.png', scale=1)

    # if self.features_opt == 'varma':
    #     # varma fixing text
    #     dtest = pd.DataFrame(dataset['1'].str.replace('[', ''))
    #     dtest = pd.DataFrame(dtest['1'].str.replace('[', ''))
    #     df_features = pd.DataFrame(dtest['1'].str.split(expand=True))
