import stumpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    your_time_series = np.random.rand(3, 100)
    window_size = 5  # Approximately, how many data points might be found in a pattern

    mps, indices = stumpy.mstump(your_time_series, m=window_size)
    motifs_idx = np.argmax(mps, axis=1)
    nn_idx = indices[np.arange(len(motifs_idx)), motifs_idx]

    fig, axs = plt.subplots(mps.shape[0] * 2, sharex=True, gridspec_kw={'hspace': 0})

    for k in range(3):
        axs[k].set_ylabel(k, fontsize='20')
        axs[k].plot(your_time_series[k])
        axs[k].set_xlabel('Time', fontsize='20')

        axs[k + mps.shape[0]].set_ylabel(k, fontsize='20')
        axs[k + mps.shape[0]].plot(mps[k], c='orange')
        axs[k + mps.shape[0]].set_xlabel('Time', fontsize='20')

        axs[k].axvline(x=motifs_idx[1], linestyle="dashed", c='black')
        axs[k].axvline(x=nn_idx[1], linestyle="dashed", c='black')
        axs[k + mps.shape[0]].axvline(x=motifs_idx[1], linestyle="dashed", c='black')
        axs[k + mps.shape[0]].axvline(x=nn_idx[1], linestyle="dashed", c='black')

        if k != 2:
            axs[k].plot(range(motifs_idx[k], motifs_idx[k] + window_size), your_time_series[k][motifs_idx[k]: motifs_idx[k] + window_size],
                        c='red', linewidth=4)
            axs[k].plot(range(nn_idx[k], nn_idx[k] + window_size), your_time_series[k][nn_idx[k]: nn_idx[k] + window_size], c='red',
                        linewidth=4)
            axs[k + mps.shape[0]].plot(motifs_idx[k], mps[k, motifs_idx[k]] + 1, marker="v", markersize=10, color='red')
            axs[k + mps.shape[0]].plot(nn_idx[k], mps[k, nn_idx[k]] + 1, marker="v", markersize=10, color='red')
        else:
            axs[k + mps.shape[0]].plot(motifs_idx[k], mps[k, motifs_idx[k]] + 1, marker="v", markersize=10,
                                       color='black')
            axs[k + mps.shape[0]].plot(nn_idx[k], mps[k, nn_idx[k]] + 1, marker="v", markersize=10, color='black')

    plt.show()

