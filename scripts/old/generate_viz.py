import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as img
from coactivation.dataset_definition import *

import warnings
import math
from scipy import ndimage

emgs = np.char.add('emg', np.arange(0, 8).astype(str))

def visualize_all_subjects(path, save_folder_name):
    base_dir = 'plots'
    save_dir = os.path.join(base_dir, f"{save_folder_name}")
    # Ensure base directory exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    # Create or clear the specific save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Adjust figure size and use gridspec for uneven subplot heights
    fig, axs = plt.subplots(2, 1, figsize=(20, 15), gridspec_kw={'height_ratios': [3, 1]})
    # EMG Signals 
    df = pd.read_csv(path)
    gt_sequence = df['gt']
    df = df[['emg0', 'emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7']]
    print(path)

    filtered_data = df.apply(lambda x: medfilt(x, kernel_size=5))
    clipped_data = filtered_data.clip(0, 1500).to_numpy().transpose()
    for channel in range(8):
        axs[0].plot(clipped_data[channel], label=f'Channel {channel}', alpha=0.7)
    # axs[0].set_title(f'EMG Signal - Majority Vote Applied: {majority_vote_flag}')
    axs[0].set_ylabel('Value - EMG Space')
    axs[0].legend(loc='upper right')
    # GT Sequence
    axs[1].plot(gt_sequence, label='GT Sequence', color='green', alpha=0.7)
    axs[1].set_xlabel('Timestep')
    axs[1].set_ylabel('GT Value')
    axs[1].legend(loc='upper right')
    axs[1].set_ylim(0, 2)
    filename = f"{path.rsplit('/', 1)[-1]}.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


def preprocess_data(path):
    """
    """
    df = pd.read_csv(path, index_col=0)
    # clip
    df[emgs] = df[emgs].clip(0, 1500)
    # normalization
    df[emgs] = df[emgs]/1500
    return df

def df_to_coactivation_map(df, metric='mean'):
    df_melt = df[np.append(['gt'], emgs)].melt(id_vars='gt',value_vars=emgs,var_name='emg')
    coactivation_map = df_melt.groupby(['gt','emg']).apply(metric).reset_index().pivot(index='gt', columns='emg', values='value')
    return coactivation_map

def plot_coactivation_map(coactivation_map, metric='median', path='', save_folder_name=''):
    base_dir = 'plots'
    save_dir = os.path.join(base_dir, f"{save_folder_name}")
    cmap_dict = {
        'median': 'Blues',
        'mean':'Purples',
        'std':'Oranges',
        'var':'Oranges',
    }
    
    fig, ax = plt.subplots(figsize=(10,3))
    im = ax.imshow(coactivation_map, cmap=cmap_dict[metric])
    ax.set_xticks(np.arange(len(emgs)), labels=emgs)
    ax.set_yticks([0,1,2], labels=['relax','open','close'])
    ax.set_title(path)
    fig.colorbar(im, ax=ax)
    if path=='':
        return
    filename = f"{path.rsplit('/', 1)[-1]}-map.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


def generate_coactivation_map(path, save_folder_name):
    df = preprocess_data(path)
    cm = df_to_coactivation_map(df, metric='mean')
    plot_coactivation_map(cm, path, save_folder_name)

def medfilt(volume, kernel_size=None):
    """
    Perform a median filter on an N-dimensional array.

    Apply a median filter to the input array using a local window-size
    given by `kernel_size`. The array will automatically be zero-padded.

    Parameters
    ----------
    volume : array_like
        An N-dimensional input array.
    kernel_size : array_like, optional
        A scalar or an N-length list giving the size of the median filter
        window in each dimension.  Elements of `kernel_size` should be odd.
        If `kernel_size` is a scalar, then this scalar is used as the size in
        each dimension. Default size is 3 for each dimension.

    Returns
    -------
    out : ndarray
        An array the same size as input containing the median filtered
        result.

    Warns
    -----
    UserWarning
        If array size is smaller than kernel size along any dimension

    See Also
    --------
    scipy.ndimage.median_filter
    scipy.signal.medfilt2d

    Notes
    -----
    The more general function `scipy.ndimage.median_filter` has a more
    efficient implementation of a median filter and therefore runs much faster.

    For 2-dimensional images with ``uint8``, ``float32`` or ``float64`` dtypes,
    the specialised function `scipy.signal.medfilt2d` may be faster.

    """
    volume = np.atleast_1d(volume)
    if not (np.issubdtype(volume.dtype, np.integer) 
            or volume.dtype in [np.float32, np.float64]):
        raise ValueError(f"dtype={volume.dtype} is not supported by medfilt")

    if kernel_size is None:
        kernel_size = [3] * volume.ndim
    kernel_size = np.asarray(kernel_size)
    if kernel_size.shape == ():
        kernel_size = np.repeat(kernel_size.item(), volume.ndim)

    for k in range(volume.ndim):
        if (kernel_size[k] % 2) != 1:
            raise ValueError("Each element of kernel_size should be odd.")
    if any(k > s for k, s in zip(kernel_size, volume.shape)):
        warnings.warn('kernel_size exceeds volume extent: the volume will be '
                      'zero-padded.',
                      stacklevel=2)

    size = math.prod(kernel_size)
    result = ndimage.rank_filter(volume, size // 2, size=kernel_size,
                                 mode='constant')

    return result

def get_coactivation_corr(coactivation_map):
    """
    """
    label = {0:'relax',1:'open',2:'close'}
    fig, ax = plt.subplots(1,3, figsize=(20,5))
    coactivation_corr_dict={}
    for i in range(3):
        data = coactivation_map.loc[i].values
        df_corr = pd.DataFrame(data[:, np.newaxis] - data)
        df_corr = 1 - np.abs(df_corr)
        coactivation_corr_dict[label[i]] = df_corr
        im = ax[i].imshow(df_corr, cmap='Greens', vmin=np.min(df_corr))
        ax[i].set_xticks(np.arange(len(emgs)), labels=emgs)
        ax[i].set_yticks(np.arange(len(emgs)), labels=emgs)
        ax[i].set_title(label[i])
        fig.colorbar(im, ax=ax[i])
    return coactivation_corr_dict

def get_coactivation_difference(map1, map2):
    """
    """
    diff_map = map1 - map2
    fig, ax = plt.subplots(figsize=(10,3))
    im = ax.imshow(diff_map, cmap='PuOr', vmin=-.2, vmax=.2)
    ax.set_xticks(np.arange(len(emgs)), labels=emgs)
    ax.set_yticks([0,1,2], labels=['relax','open','close'])
    fig.colorbar(im, ax=ax)
    return diff_map

if __name__ == "__main__":
    data_dir = 'collected_data'
    for root, subdirs, files in os.walk(data_dir):
        print(root)
        if root != data_dir and root[15] == '2':
            print(f"\nIn subdirectory: {root}")
            if (int(root[15:19]) > 2022) and files: 
                print("Files:")
                for file in files:
                    if file[-4:] == '.csv':
                        print(file)
                        print(os.path.join(f"{root}", file))
                        visualize_all_subjects(os.path.join(f"{root}", file), root[15:])
                        generate_coactivation_map(os.path.join(f"{root}", file), root[15:])
                        

