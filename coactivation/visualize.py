import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from coactivation.process import preprocess_emgs, get_stratified_cm

emgs = np.char.add('emg', np.arange(0, 8).astype(str))

def visualize_emgs(path):
    '''plot raw emg signals and gt sequence from csv file

    parameters
    ----------
    path : str
        path to the csv file containing 'gt' column and emg0-7 columns

    returns
    -------
    None
    '''
    # Adjust figure size and use gridspec for uneven subplot heights
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    # EMG Signals 
    df = pd.read_csv(path)
    gt_sequence = df['gt']
    df = df[emgs]
    plot_data = df.to_numpy().transpose()
    for channel in range(8):
        axs[0].plot(plot_data[channel], label=f'Channel {channel}', alpha=0.7) 
    axs[0].set_ylabel('Value - EMG Space')
    axs[0].legend(loc='upper right')
    # GT Sequence
    axs[1].plot(gt_sequence, label='GT Sequence', color='green', alpha=0.7)
    axs[1].set_xlabel('Timestep')
    axs[1].set_ylabel('GT Value')
    axs[1].legend(loc='upper right')
    axs[1].set_ylim(0, 2)

def visualize_stratified_median_emgs(path, hz=50):
    '''plot median emg signals stratified by gt windows and highlight segments

    parameters
    ----------
    path : str
        path to the csv file for preprocessing and plotting
    hz : int, optional
        sampling frequency for normalization (default: 50)

    returns
    -------
    None
    '''
    df_preprocessed = preprocess_emgs(path, hz=hz)
    gt_sequence = df_preprocessed['gt']
    df_preprocessed['window'] = np.cumsum(np.diff(df_preprocessed['gt'], prepend=0)!=0)
    plot_data = pd.merge(df_preprocessed[['window']], get_stratified_cm(df_preprocessed), on = 'window', how = 'left')[emgs].to_numpy().transpose()

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    for channel in range(8):
        axs[0].plot(plot_data[channel], label=f'Channel {channel}', alpha=0.7) 
    axs[0].set_ylabel('Value - EMG Space')
    axs[0].legend(loc='upper right')
    # GT Sequence
    axs[1].plot(gt_sequence, label='GT Sequence', color='green', alpha=0.7)
    axs[1].set_xlabel('Timestep')
    axs[1].set_ylabel('GT Value')
    axs[1].legend(loc='upper right')
    axs[1].set_ylim(0, 2)

    colors = {0: 'red', 1: 'blue', 2: 'green'}
    prev_idx = 0
    prev_value = gt_sequence.iloc[0]

    for idx in range(1, len(gt_sequence)):
        if gt_sequence.iloc[idx] != prev_value or idx == len(gt_sequence) - 1:
            axs[0].axvspan(prev_idx, idx, color=colors[prev_value], alpha=0.05)
            axs[1].axvspan(prev_idx, idx, color=colors[prev_value], alpha=0.05)
            prev_idx = idx
            prev_value = gt_sequence.iloc[idx]


def visualize_stratified_median_emgs2(path, hz=50):
    '''plot stratified median emg signals with direct grouping by window

    parameters
    ----------
    path : str
        path to the csv file for preprocessing and plotting
    hz : int, optional
        sampling frequency for normalization (default: 50)

    returns
    -------
    None
    '''
    df_preprocessed = preprocess_emgs(path, hz=hz)
    plot_data = get_stratified_cm(df_preprocessed)
    gt_sequence = df_preprocessed['gt']
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    for channel in range(8):
        axs[0].plot(plot_data[channel], label=f'Channel {channel}', alpha=0.7) 
    axs[0].set_ylabel('Value - EMG Space')
    axs[0].legend(loc='upper right')
    # GT Sequence
    axs[1].plot(gt_sequence, label='GT Sequence', color='green', alpha=0.7)
    axs[1].set_xlabel('Timestep')
    axs[1].set_ylabel('GT Value')
    axs[1].legend(loc='upper right')
    axs[1].set_ylim(0, 2)

    colors = {0: 'red', 1: 'blue', 2: 'green'}
    prev_idx = 0
    prev_value = gt_sequence.iloc[0]

    for idx in range(1, len(gt_sequence)):
        if gt_sequence.iloc[idx] != prev_value or idx == len(gt_sequence) - 1:
            axs[0].axvspan(prev_idx, idx, color=colors[prev_value], alpha=0.05)
            axs[1].axvspan(prev_idx, idx, color=colors[prev_value], alpha=0.05)
            prev_idx = idx
            prev_value = gt_sequence.iloc[idx]


def plot_coactivation_map(coactivation_map, metric='median', title='', vmax=-1):
    '''plot a coactivation heatmap for given metric and title

    parameters
    ----------
    coactivation_map : pd.DataFrame
        pivoted coactivation matrix with index 'gt' and emg columns
    metric : str, optional
        aggregation metric used ('median' or 'var')
    title : str, optional
        title suffix for the plot
    vmax : float, optional
        maximum value for color scale (default: autoset)

    returns
    -------
    None
    '''
    cmap_dict = {
        'median': 'Blues',
        'var':'Oranges',
    }
    fig, ax = plt.subplots(figsize=(10,3))
    if vmax==-1:
        vmax=0.25
    if (vmax>0.25) or (vmax==0):
        vmax=np.max(coactivation_map.values)
    plot = np.zeros((3, 8))
    plot[coactivation_map.reset_index()['gt'].values, :] = coactivation_map.values

    im = ax.imshow(plot, cmap=cmap_dict[metric], vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(len(emgs)), labels=emgs)
    ax.set_yticks([0,1,2], labels=['relax','open','close'])
    ax.set_title('Coactivation Map (' + metric + '): ' + title)
    fig.colorbar(im, ax=ax)


def plot_one_class_coactivation_map(coactivation_map, metric='median', title='', label='', cmap='Blues'):
    '''plot a single-row coactivation heatmap with class label

    parameters
    ----------
    coactivation_map : array-like
        1d coactivation values for one class
    metric : str, optional
        aggregation metric used ('median' or 'var')
    title : str, optional
        title suffix for the plot
    label : str, optional
        y-axis label for the row
    cmap : str, optional
        colormap name for the plot

    returns
    -------
    None
    '''
    fig, ax = plt.subplots(figsize=(10,1))
    vmax=0.5
    if metric=='var':
        vmax=np.max(coactivation_map)
    im = ax.imshow([coactivation_map], cmap=cmap,
                   vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(len(emgs)), labels=emgs)
    ax.set_yticks([0], labels=[label])
    ax.set_title('Coactivation Map (' + metric + '): ' + title)
    fig.colorbar(im, ax=ax)


def plot_coactivation_map_with_histograms(identities, main_identity, metric='median', title=''):
    '''plot coactivation histograms per cell against a main identity

    parameters
    ----------
    identities : list of 2d arrays
        list of coactivation matrices to compare
    main_identity : 2d array
        reference coactivation matrix for coloring
    metric : str, optional
        aggregation metric used ('median' or 'var')
    title : str, optional
        title suffix for the plot

    returns
    -------
    None
    '''
    fig = plt.figure(figsize=(12, 4))
    outer_gs = gridspec.GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(outer_gs[0, 0])
    ax.set_title('Coactivation Map (' + metric + '): ' + title)
    ax.axis("off") 

    gs = gridspec.GridSpecFromSubplotSpec(3, 8, subplot_spec=ax.get_subplotspec(), wspace=0.4, hspace=0.4)
    norm = Normalize(vmin=np.min(main_identity), vmax=np.max(main_identity))
    cmap = cm.Blues 

    for row in range(3):
        for col in range(8):
            actual_values = [identity[row, col] for identity in identities]
            deviations = [value - main_identity[row, col] for value in actual_values]
            counts, bins = np.histogram(deviations, bins=10)
            
            cell_ax = fig.add_subplot(gs[row, col])
            for i in range(len(bins) - 1):
                bin_midpoint = main_identity[row, col] + (bins[i] + bins[i + 1]) / 2
                bin_color = cmap(norm(bin_midpoint))
                cell_ax.bar(bins[i], counts[i], width=(bins[i + 1] - bins[i]), color=bin_color, edgecolor="none", align="edge")
            
            cell_ax.axvline(0, color="red", linestyle="--", linewidth=1)
            cell_ax.set_xlim(-1, 1) 
            
            cell_ax.set_xticks([])
            cell_ax.set_yticks([])
            cell_ax.spines['top'].set_visible(False)
            cell_ax.spines['right'].set_visible(False)
            cell_ax.spines['left'].set_visible(False)
            cell_ax.spines['bottom'].set_visible(False)

    row_labels = ['relax', 'open', 'close']
    col_labels = [f'emg{i}' for i in range(8)]
    
    for row, label in enumerate(row_labels):
        fig.text(0.12, 0.75 - (row * 0.28), label, va='center', ha='center', fontsize=10, rotation='vertical')

    for col, label in enumerate(col_labels):
        fig.text(0.16 + (col * 0.1), 0.08, label, va='center', ha='center', fontsize=10)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.02, label='coactivation mean')

    plt.show()


#======= Discard Pile ========#
'''
def plot_coactivation_corr(coactivation_map):
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

def plot_bool_map(boolean_map, title=''):
    colors=['white','darkblue']
    cmap=mcolors.ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(10,3))
    ax.imshow(boolean_map, cmap=cmap)
    ax.set_xticks(np.arange(len(emgs)), labels=emgs)
    ax.set_yticks([0,1,2], labels=['relax','open','close'])
    ax.set_title('Boolean Coactivation Map: ' + title)
    ax.grid(which="minor", color='black', linestyle='-', linewidth=0.5)
    legend_labels = [mpatches.Patch(color=colors[0], label='Deactivated'),
                     mpatches.Patch(color=colors[1], label='Activated')]
    plt.legend(handles=legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()

def plot_alignment_map(alignment_map, title1='', title2=''):
    colors=['white','forestgreen','deepskyblue','indianred']
    cmap=mcolors.ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(10,3))
    ax.imshow(alignment_map, cmap=cmap)
    ax.set_xticks(np.arange(len(emgs)), labels=emgs)
    ax.set_yticks([0,1,2], labels=['relax','open','close'])
    ax.set_title('Alignment Coactivation Map: ' + title1 + ' on ' + title2)
    ax.grid(which="minor", color='black', linestyle='-', linewidth=0.5)
    legend_labels = [mpatches.Patch(color=colors[0], label='0, 0'),
                     mpatches.Patch(color=colors[1], label='1, 1'),
                     mpatches.Patch(color=colors[2], label='0, 1'),
                     mpatches.Patch(color=colors[3], label='1, 0'),]
    plt.legend(handles=legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    return
'''
#==============================#