import stumpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from river import drift
from mindarmour import ConceptDriftCheckTimeSeries


def diff_cog(x):
    diff_x = x.diff()
    diff_x[diff_x > 180] = diff_x[diff_x > 180] - 360
    diff_x[diff_x < -180] = diff_x[diff_x < -180] + 360
    return diff_x


if __name__ == "__main__":

    data = pd.read_csv("traj1.csv")
    series = data[['cog', 'sog']]
    # series['sog'] = series['sog'].rolling(5).mean()
    # series['sog'] = series['sog'].diff()
    data['dtime'] = data['time'].astype('datetime64[ns]').diff()
    series = series.fillna(0)
    # series['rcog'] = diff_cog(series['cog'])
    your_time_series = series.T.to_numpy()
    your_time_series = (your_time_series - your_time_series.min()) / (your_time_series.max() - your_time_series.min())

    # your_time_series = np.random.rand(3, 100)
    window_size = 200  # Approximately, how many data points might be found in a pattern

    mps, indices = stumpy.mstump(your_time_series, m=window_size)

    df = pd.DataFrame(mps).T
    idx = {}
    for i in range(mps.shape[0]):
        a = df[i].diff()
        b = a.diff()
        idx[i] = b[b > 0.3].index

    motifs_idx = np.argmin(mps, axis=1)
    nn_idx = indices[np.arange(len(motifs_idx)), motifs_idx]

### ADWIN
    idx_adwin = {}
    var_adwin = {}
    for k in range(mps.shape[0]):
        drift_detector = drift.ADWIN(delta=0.002)
        # drift_detector = drift.KSWIN(window_size=500)
        min_w = len(your_time_series[k])
        drifts1 = []
        drifts1_var = []
        for i, val in enumerate(your_time_series[k]):
            drift_detector.update(val)   # Data is processed one sample at a time
            drifts1_var.append(drift_detector.variance)
            if drift_detector.change_detected:
               drifts1.append(i)
               # if min_w > drift_detector.width:
               #     min_w = drift_detector.width
            # drift_detector.reset()
        idx_adwin[k] = drifts1
        var_adwin[k] = drifts1_var
        print(drifts1)

    test = pd.DataFrame(idx_adwin[0])
    all_idx_adwin = test[test.isin(idx_adwin[1]).values]
    if len(all_idx_adwin) > 0:
        all_idx_adwin = all_idx_adwin[0].to_list()

#### concept drift
    idx_cd = {}
    drift_score={}
    for k in range(mps.shape[0]):
        concept = ConceptDriftCheckTimeSeries(window_size=100, rolling_window=10, step=10, threshold_index=1.5,
                                              need_label=False)
        drift_score[k], threshold, idx_cd[k] = concept.concept_check(your_time_series[k])


### image

    fig, axs = plt.subplots(mps.shape[0] * 2, sharex=True, gridspec_kw={'hspace': 0})

    for k in range(mps.shape[0]):
        axs[k].set_ylabel(k, fontsize='20')
        axs[k].plot(your_time_series[k])
        axs[k].set_xlabel('Time', fontsize='20')

        axs[k+ mps.shape[0]].set_ylabel(k, fontsize='20')
        axs[k+ mps.shape[0]].plot(var_adwin[k])
        axs[k+ mps.shape[0]].set_xlabel('Time', fontsize='20')

        # axs[k + mps.shape[0]].set_ylabel(k, fontsize='20')
        # axs[k + mps.shape[0]].plot(mps[k], c='orange')
        # axs[k + mps.shape[0]].set_xlabel('Time', fontsize='20')

        # axs[k].axvline(x=motifs_idx[1], linestyle="dashed", c='black')
        # axs[k].axvline(x=nn_idx[1], linestyle="dashed", c='black')
        axs[k + mps.shape[0]].axvline(x=motifs_idx[1], linestyle="dashed", c='black')
        axs[k + mps.shape[0]].axvline(x=nn_idx[1], linestyle="dashed", c='black')

        for j in range(len(idx_adwin[k])):
            axs[k].axvline(x=idx_adwin[k][j], linestyle="dashed", c='black')
        for j in range(len(all_idx_adwin)):
            axs[k].axvline(x=all_idx_adwin[j], linestyle="dashed", c='red')
        # for j in range(len(idx_cd[k])):
        #     axs[k+ mps.shape[0]].axvline(x=idx_cd[k][j], linestyle="dashed", c='blue')

        # if k != 2:
        #     axs[k].plot(range(motifs_idx[k], motifs_idx[k] + window_size), your_time_series[k][motifs_idx[k]: motifs_idx[k] + window_size],
        #                 c='red', linewidth=4)
        #     axs[k].plot(range(nn_idx[k], nn_idx[k] + window_size), your_time_series[k][nn_idx[k]: nn_idx[k] + window_size], c='red',
        #                 linewidth=4)
        #     axs[k + mps.shape[0]].plot(motifs_idx[k], mps[k, motifs_idx[k]] + 1, marker="v", markersize=10, color='red')
        #     axs[k + mps.shape[0]].plot(nn_idx[k], mps[k, nn_idx[k]] + 1, marker="v", markersize=10, color='red')
        # else:
        #     axs[k + mps.shape[0]].plot(motifs_idx[k], mps[k, motifs_idx[k]] + 1, marker="v", markersize=10,
        #                                color='black')
        #     axs[k + mps.shape[0]].plot(nn_idx[k], mps[k, nn_idx[k]] + 1, marker="v", markersize=10, color='black')

    plt.show()

