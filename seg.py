import stumpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from river import drift
from mindarmour import ConceptDriftCheckTimeSeries
from haversine import haversine


def diff_cog(x):
    diff_x = x.diff()
    diff_x[diff_x > 180] = diff_x[diff_x > 180] - 360
    diff_x[diff_x < -180] = diff_x[diff_x < -180] + 360
    return diff_x


def my_dist(x):
    p1, p2 = x
    return haversine(p1, p2)


def speed_lat_lon(x):
    new_data = pd.DataFrame()
    new_data['position'] = list(zip(x['lat'],x['lon']))
    new_data['dtime'] = x['dtime'].apply(lambda x: x.seconds)
    new_data['position_next'] = new_data['position'].shift(-1)
    new_data['dtime'] = new_data['dtime'].shift(-1)
    new_data['comb_position'] = list(zip(new_data['position'], new_data['position_next']))
    new_data = new_data.iloc[:-1, :]

    new_data['hav_dist'] = new_data['comb_position'].apply(lambda x: my_dist(x))
    speed = new_data['hav_dist']/new_data['dtime']

    return speed



if __name__ == "__main__":

    data = pd.read_csv("traj1.csv")
    data['dtime'] = data['time'].astype('datetime64[ns]').diff()

    speed_series = speed_lat_lon(data)
    series = data[['cog', 'sog']]
    # series['sog'] = series['sog'].rolling(5).mean()
    # series['sog'] = series['sog'].diff()
    series = series.fillna(0)
    # series['rcog'] = diff_cog(series['cog'])
    your_time_series = series.T.to_numpy()
    your_time_series = (your_time_series - your_time_series.min()) / (your_time_series.max() - your_time_series.min())

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(data['sog'])
    ax[0].set_title("SOG")
    ax[1].plot(speed_series)
    ax[1].set_title("speed based on Lat and Lon")
    plt.show()

    window_size = 20  # Approximately, how many data points might be found in a pattern
    n_segments = 30   # number of segments to divide the trajectory

    mps_multi, indices_multi = stumpy.mstump(your_time_series, m=window_size)
    # mps2 = stumpy.stump(your_time_series[0], m=window_size)
    cac_multi, regime_locations_multi = stumpy.fluss(indices_multi[0,:], L=window_size, n_regimes=n_segments, excl_factor=1)

    mps_sog = stumpy.stump(data['sog'], m=window_size)
    cac_sog, regime_locations_sog = stumpy.fluss(mps_sog[:, 1], L=window_size, n_regimes=n_segments, excl_factor=1)

    mps_sog_norm = stumpy.stump(your_time_series[1], m=window_size)
    cac_sog_norm, regime_locations_sog_norm = stumpy.fluss(mps_sog_norm[:, 1], L=window_size, n_regimes=n_segments, excl_factor=1)

    mps_speed = stumpy.stump(speed_series, m=window_size)
    cac_speed, regime_locations_speed = stumpy.fluss(mps_speed[:, 1], L=window_size, n_regimes=n_segments, excl_factor=1)

    mps_cog = stumpy.stump(data['cog'], m=window_size)
    cac_cog, regime_locations_cog = stumpy.fluss(mps_cog[:, 1], L=window_size, n_regimes=n_segments, excl_factor=1)

    mps_cog_norm = stumpy.stump(your_time_series[0], m=window_size)
    cac_cog_norm, regime_locations_cog_norm = stumpy.fluss(mps_cog_norm[:, 1], L=window_size, n_regimes=n_segments, excl_factor=1)


    # stream = stumpy.floss(mps, speed_series, m=window_size, L=window_size, excl_factor=1)
    # windows = []
    # for i, t in enumerate(your_time_series[0]):
    #     stream.update(t)
    #
    #     if i % 100 == 0:
    #         windows.append((stream.T_, stream.cac_1d_))

    init = 0
    seg = 0
    regime_locations_multi.sort()
    regime_locations_multi = np.append(regime_locations_multi,len(your_time_series[0]))
    for k in regime_locations_multi:
        data.loc[init:k, 'segmentation_multi'] = seg
        seg = seg + 1
        init = k

    init = 0
    seg = 0
    regime_locations_sog.sort()
    regime_locations_sog = np.append(regime_locations_sog,len(data['sog']))
    for k in regime_locations_sog:
        data.loc[init:k, 'segmentation_sog'] = seg
        seg = seg + 1
        init = k

    init = 0
    seg = 0
    regime_locations_sog_norm.sort()
    regime_locations_sog_norm = np.append(regime_locations_sog_norm,len(your_time_series[1]))
    for k in regime_locations_sog_norm:
        data.loc[init:k, 'segmentation_sog_norm'] = seg
        seg = seg + 1
        init = k

    init = 0
    seg = 0
    regime_locations_cog.sort()
    regime_locations_cog = np.append(regime_locations_cog,len(data['cog']))
    for k in regime_locations_cog:
        data.loc[init:k, 'segmentation_cog'] = seg
        seg = seg + 1
        init = k

    init = 0
    seg = 0
    regime_locations_cog_norm.sort()
    regime_locations_cog_norm = np.append(regime_locations_cog_norm,len(your_time_series[0]))
    for k in regime_locations_cog_norm:
        data.loc[init:k, 'segmentation_cog_norm'] = seg
        seg = seg + 1
        init = k

    init = 0
    seg = 0
    regime_locations_speed.sort()
    regime_locations_speed = np.append(regime_locations_speed,len(speed_series))
    for k in regime_locations_speed:
        data.loc[init:k, 'segmentation_speed'] = seg
        seg = seg + 1
        init = k
    data.to_csv('traj1_seg.csv')

    fig, axs = plt.subplots(3, sharex=True, gridspec_kw={'hspace': 0})
    axs[0].set_title('Multi')
    axs[0].plot(range(your_time_series[0].shape[0]), your_time_series[0])
    axs[1].plot(range(your_time_series[1].shape[0]), your_time_series[1])
    axs[2].plot(range(cac_multi.shape[0]), cac_multi, color='C1')
    # axs[1].plot(range(stream.cac_1d_.shape[0]), stream.cac_1d_, color='green')
    for i in regime_locations_multi:
        axs[0].axvline(x=i, linestyle="dashed")
        axs[1].axvline(x=i, linestyle="dashed")
        axs[2].axvline(x=i, linestyle="dashed")
    plt.show()

    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    axs[0].set_title('SOG')
    axs[0].plot(range(data['sog'].shape[0]), data['sog'])
    axs[1].plot(range(cac_sog.shape[0]), cac_sog, color='C1')
    # axs[1].plot(range(stream.cac_1d_.shape[0]), stream.cac_1d_, color='green')
    for i in regime_locations_sog:
        axs[0].axvline(x=i, linestyle="dashed")
        axs[1].axvline(x=i, linestyle="dashed")
    plt.show()

    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    axs[0].set_title('SOG norm')
    axs[0].plot(range(your_time_series[1].shape[0]), your_time_series[1])
    axs[1].plot(range(cac_sog_norm.shape[0]), cac_sog_norm, color='C1')
    # axs[1].plot(range(stream.cac_1d_.shape[0]), stream.cac_1d_, color='green')
    for i in regime_locations_sog_norm:
        axs[0].axvline(x=i, linestyle="dashed")
        axs[1].axvline(x=i, linestyle="dashed")
    plt.show()

    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    axs[0].set_title('Speed')
    axs[0].plot(range(speed_series.shape[0]), speed_series)
    axs[1].plot(range(cac_speed.shape[0]), cac_speed, color='C1')
    # axs[1].plot(range(stream.cac_1d_.shape[0]), stream.cac_1d_, color='green')
    for i in regime_locations_speed:
        axs[0].axvline(x=i, linestyle="dashed")
        axs[1].axvline(x=i, linestyle="dashed")
    plt.show()

    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    axs[0].set_title('COG')
    axs[0].plot(range(data['cog'].shape[0]), data['cog'])
    axs[1].plot(range(cac_cog.shape[0]), cac_cog, color='C1')
    # axs[1].plot(range(stream.cac_1d_.shape[0]), stream.cac_1d_, color='green')
    for i in regime_locations_cog:
        axs[0].axvline(x=i, linestyle="dashed")
        axs[1].axvline(x=i, linestyle="dashed")
    plt.show()

    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    axs[0].set_title('COG norm')
    axs[0].plot(range(your_time_series[0].shape[0]), your_time_series[0])
    axs[1].plot(range(cac_cog_norm.shape[0]), cac_cog_norm, color='C1')
    # axs[1].plot(range(stream.cac_1d_.shape[0]), stream.cac_1d_, color='green')
    for i in regime_locations_cog_norm:
        axs[0].axvline(x=i, linestyle="dashed")
        axs[1].axvline(x=i, linestyle="dashed")
    plt.show()


    df = pd.DataFrame(mps_multi).T
    idx = {}
    for i in range(mps_multi.shape[0]):
        a = df[i].diff()
        b = a.diff()
        idx[i] = b[b > 0.3].index

    motifs_idx = np.argmin(mps_multi, axis=1)
    nn_idx = indices_multi[np.arange(len(motifs_idx)), motifs_idx]



### image
    #
    # fig, axs = plt.subplots(mps_multi.shape[0] * 2, sharex=True, gridspec_kw={'hspace': 0})
    #
    # for k in range(mps_multi.shape[0]):
    #     axs[k].set_ylabel(k, fontsize='20')
    #     axs[k].plot(your_time_series[k])
    #     axs[k].set_xlabel('Time', fontsize='20')
    #
    #     axs[k + mps_multi.shape[0]].set_ylabel(k, fontsize='20')
    #     axs[k + mps_multi.shape[0]].plot(mps_multi[k], c='orange')
    #     axs[k + mps_multi.shape[0]].set_xlabel('Time', fontsize='20')
    #
    #     axs[k].axvline(x=motifs_idx[1], linestyle="dashed", c='black')
    #     axs[k].axvline(x=nn_idx[1], linestyle="dashed", c='black')
    #     axs[k + mps_multi.shape[0]].axvline(x=motifs_idx[1], linestyle="dashed", c='black')
    #     axs[k + mps_multi.shape[0]].axvline(x=nn_idx[1], linestyle="dashed", c='black')
    #
    #     if k != 2:
    #         axs[k].plot(range(motifs_idx[k], motifs_idx[k] + window_size), your_time_series[k][motifs_idx[k]: motifs_idx[k] + window_size],
    #                     c='red', linewidth=4)
    #         axs[k].plot(range(nn_idx[k], nn_idx[k] + window_size), your_time_series[k][nn_idx[k]: nn_idx[k] + window_size], c='red',
    #                     linewidth=4)
    #         axs[k + mps_multi.shape[0]].plot(motifs_idx[k], mps_multi[k, motifs_idx[k]] + 1, marker="v", markersize=10, color='red')
    #         axs[k + mps_multi.shape[0]].plot(nn_idx[k], mps_multi[k, nn_idx[k]] + 1, marker="v", markersize=10, color='red')
    #     else:
    #         axs[k + mps_multi.shape[0]].plot(motifs_idx[k], mps_multi[k, motifs_idx[k]] + 1, marker="v", markersize=10,
    #                                    color='black')
    #         axs[k + mps_multi.shape[0]].plot(nn_idx[k], mps_multi[k, nn_idx[k]] + 1, marker="v", markersize=10, color='black')
    #
    # plt.show()

