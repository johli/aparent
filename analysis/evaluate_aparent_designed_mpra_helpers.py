import pandas as pd
import scipy
import numpy as np
import scipy.sparse as sp

from scipy.stats import pearsonr

import operator
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import matplotlib.colors as colors

import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import roc_auc_score



#Sequence Plotting Functions

def letterAt(letter, x, y, yscale=1, ax=None, color=None, alpha=1.0):

    #fp = FontProperties(family="Arial", weight="bold")
    fp = FontProperties(family="Ubuntu", weight="bold")
    globscale = 1.35
    LETTERS = {	"T" : TextPath((-0.305, 0), "T", size=1, prop=fp),
                "G" : TextPath((-0.384, 0), "G", size=1, prop=fp),
                "A" : TextPath((-0.35, 0), "A", size=1, prop=fp),
                "C" : TextPath((-0.366, 0), "C", size=1, prop=fp),
                "UP" : TextPath((-0.488, 0), '$\\Uparrow$', size=1, prop=fp),
                "DN" : TextPath((-0.488, 0), '$\\Downarrow$', size=1, prop=fp),
                "(" : TextPath((-0.25, 0), "(", size=1, prop=fp),
                "." : TextPath((-0.125, 0), "-", size=1, prop=fp),
                ")" : TextPath((-0.1, 0), ")", size=1, prop=fp)}
    COLOR_SCHEME = {'G': 'orange', 
                    'A': 'red', 
                    'C': 'blue', 
                    'T': 'darkgreen',
                    'UP': 'green', 
                    'DN': 'red',
                    '(': 'black',
                    '.': 'black', 
                    ')': 'black'}


    text = LETTERS[letter]

    chosen_color = COLOR_SCHEME[letter]
    if color is not None :
        chosen_color = color

    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    p = PathPatch(text, lw=0, fc=chosen_color, alpha=alpha, transform=t)
    if ax != None:
        ax.add_artist(p)
    return p

#PWM Helper Functions

def hamming_distance(seq1, seq2) :
    dist = 0
    for j in range(0, len(seq1)) :
        if seq1[j] != seq2[j] :
            dist += 1
    
    return dist

def get_pwm(seqs) :
    pwm = np.zeros((len(seqs[0]), 4))
    
    for i in range(0, len(seqs)) :
        seq = seqs[i]
        for j in range(0, len(seq)) :
            if seq[j] == 'A' :
                pwm[j, 0] += 1
            elif seq[j] == 'C' :
                pwm[j, 1] += 1
            elif seq[j] == 'G' :
                pwm[j, 2] += 1
            elif seq[j] == 'T' :
                pwm[j, 3] += 1
    
    for j in range(0, pwm.shape[0]) :
        pwm[j, :] /= np.sum(pwm[j, :])
    
    return pwm

def one_hot_decode(one_hot) :
    seq = ''
    for j in range(0, one_hot.shape[0]) :
        if one_hot[j, 0] == 1 :
            seq += 'A'
        elif one_hot[j, 1] == 1 :
            seq += 'C'
        elif one_hot[j, 2] == 1 :
            seq += 'G'
        elif one_hot[j, 3] == 1 :
            seq += 'T'
        else :
            seq += 'N'
    return seq

def get_consensus(pwm) :
    one_hot = np.zeros(pwm.shape)
    for j in range(0, pwm.shape[0]) :
        if np.sum(pwm[j, :]) > 0.0 :
            max_k = np.argmax(pwm[j, :])
            one_hot[j, max_k] = 1
    return one_hot

def find_wt_pwm(wt_seq, pwms) :
    min_i = 0
    min_dist = 30
    for i in range(0, pwms.shape[0]) :
        consensus_seq = one_hot_decode(get_consensus(pwms[i, :, :]))
        dist = hamming_distance(consensus_seq[:163], wt_seq[1:])
        if dist < min_dist :
            min_i = i
            min_dist = dist
    
    return pwms[min_i, :, :]

def find_wt_yhat(wt_seq, pwms, cuts) :
    min_i = 0
    min_dist = 30
    for i in range(0, pwms.shape[0]) :
        consensus_seq = one_hot_decode(get_consensus(pwms[i, :, :]))
        dist = hamming_distance(consensus_seq[:163], wt_seq[1:])
        if dist < min_dist :
            min_i = i
            min_dist = dist
    
    return cuts[min_i, :]

def aggregate_and_append_predictions(seq_df, seq_cuts, pred_df, cuts_pred) :
    #Aggregate predictions over barcoded replicates

    pred_df['row_index'] = np.arange(len(pred_df), dtype=np.int)
    pred_df['iso_pred_from_cuts'] = np.sum(np.array(cuts_pred[:, 77:107].todense()), axis=-1)
    pred_df['iso_pred_mix'] = (pred_df['iso_pred'] + pred_df['iso_pred_from_cuts']) / 2.

    pred_df_group = pred_df.groupby("master_seq")
    pred_df_agg = pred_df_group.agg({
        'master_seq' : 'first',
        'row_index' : lambda x: tuple(x),
        'iso_pred' : lambda x: tuple(x),
        'iso_pred_from_cuts' : lambda x: tuple(x),
        'iso_pred_mix' : lambda x: tuple(x),
        'pooled_total_count' : lambda x: tuple(x)
    })

    pred_df_agg['sum_total_count'] = pred_df_agg['pooled_total_count'].apply(lambda x: np.sum(list(x)))

    for pred_suffix in ['', '_from_cuts', '_mix'] :
        pred_df_agg['iso_pred' + pred_suffix] = pred_df_agg.apply(
            lambda row: np.sum(np.ravel(list(row['iso_pred' + pred_suffix])) * np.ravel(list(row['pooled_total_count']))) / row['sum_total_count']
            ,axis=1
        )
        pred_df_agg['logodds_pred' + pred_suffix] = np.log(pred_df_agg['iso_pred' + pred_suffix] / (1. - pred_df_agg['iso_pred' + pred_suffix]))

    dense_cuts_pred = np.array(cuts_pred.todense())
    dense_cuts_pred_agg = np.zeros((len(pred_df_agg), dense_cuts_pred.shape[1]))

    i = 0
    for _, row in pred_df_agg.iterrows() :
        old_ix = list(row['row_index'])
        counts = np.ravel(list(row['pooled_total_count'])).reshape(-1, 1)
        total_count = row['sum_total_count']
        dense_cuts_pred_agg[i, :] = np.sum(dense_cuts_pred[old_ix, :] * counts / total_count, axis=0)
        
        i += 1

    pred_df_agg = pred_df_agg[['master_seq', 'iso_pred', 'iso_pred_from_cuts', 'iso_pred_mix', 'logodds_pred', 'logodds_pred_from_cuts', 'logodds_pred_mix']]

    #Join dataframe with prediction table and calculate true cut probabilities

    seq_df['row_index_true'] = np.arange(len(seq_df), dtype=np.int)
    pred_df_agg['row_index_pred'] = np.arange(len(pred_df_agg), dtype=np.int)

    seq_df = seq_df.join(pred_df_agg.set_index('master_seq'), on='master_seq', how='inner').copy().reset_index(drop=True)

    seq_cuts = seq_cuts[np.ravel(seq_df['row_index_true'].values), :]
    dense_cuts_pred_agg = dense_cuts_pred_agg[np.ravel(seq_df['row_index_pred'].values), 20:]

    cut_true = np.concatenate([np.array(seq_cuts[:, 180 + 20: 180 + 205].todense()), np.array(seq_cuts[:, -1].todense()).reshape(-1, 1)], axis=-1)
    #Add small pseudo count to true cuts
    cut_true += 0.0005
    cut_true = cut_true / np.sum(cut_true, axis=-1).reshape(-1, 1)

    seq_df['cut_prob_true'] = [cut_true[i, :] for i in range(len(seq_df))]
    seq_df['cut_prob_pred'] = [dense_cuts_pred_agg[i, :] for i in range(len(seq_df))]

    return seq_df

#Max Isoform Helper Functions

def plot_sequence_logo(df, df_human, max_iso_pwm_dict, gene, subexperiments, override_mean_stats=False, plot_percentile=True, plot_mean_logo=True, plot_max_logo=True, plot_actual_pwm=True, plot_opt_pwm=True, black_fixed_seq=True, max_index=None, true_column='median_proximal_vs_distal_logodds', figsize=(12, 3), width_ratios=[1, 7], logo_height=1.0, usage_unit='log', plot_snvs=False, seq_trim_start=0, seq_trim_end=164, plot_start=0, plot_end=164, pas_downscaling=1.0, save_figs=False, fig_name=None, fig_dpi=300) :

    #Make sequence logo
    
    df_seqs = df.query("variant == 'wt' or variant == 'sampled'")
    df_seqs = df_seqs.loc[df_seqs.subexperiment.isin(subexperiments)]
    
    #Mean logos
    seqs = list(df_seqs['master_seq'].values)
    
    n_seqs = len(seqs)
    pwm = get_pwm(seqs)
    
    wt_seqs = list(df_seqs['wt_seq'].unique())
    
    wt_mean_logodds = np.zeros(len(wt_seqs))
    wt_std_logodds = np.zeros(len(wt_seqs))
    
    #Get wt seq cluster statistics
    for i, wt_seq in enumerate(wt_seqs) :
        wt_mean_logodds[i] = np.mean(df_seqs.query("wt_seq == '" + wt_seq + "'")[true_column])
        wt_std_logodds[i] = np.std(df_seqs.query("wt_seq == '" + wt_seq + "'")[true_column])
    
    opt_pwm = np.zeros(pwm.shape)
    for wt_seq in wt_seqs :
        subexperiment = list(df_seqs.query("wt_seq == '" + wt_seq + "'")['subexperiment'].values)[0]
        subexp_pwm = np.vstack([np.ones((1, 4)) * 0.25, find_wt_pwm(wt_seq, max_iso_pwm_dict[gene + '_' + subexperiment])])[:164, :]
        
        opt_pwm += subexp_pwm
    
    
    fixed_seq = []
    if np.sum(opt_pwm) > 0 :
        for j in range(0, opt_pwm.shape[0]) :
            if np.sum(opt_pwm[j, :]) > 0 :
                opt_pwm[j, :] /= np.sum(opt_pwm[j, :])
            
            if np.max(opt_pwm[j, :]) == 1. :
                fixed_seq.append(True)
            else :
                fixed_seq.append(False)
    
    #Slice according to seq trim index
    seqs = [seq[seq_trim_start: seq_trim_end] for seq in seqs]
    fixed_seq = fixed_seq[seq_trim_start: seq_trim_end]
    pwm = pwm[:, seq_trim_start: seq_trim_end]
    opt_pwm = opt_pwm[:, seq_trim_start: seq_trim_end]
    
    pwm += 0.001
    for j in range(0, pwm.shape[0]) :
        pwm[j, :] /= np.sum(pwm[j, :])

    #Plot actual array pwm
    
    entropy = np.zeros(pwm.shape)
    entropy[pwm > 0] = pwm[pwm > 0] * -np.log2(pwm[pwm > 0])
    entropy = np.sum(entropy, axis=1)
    conservation = 2 - entropy

    fig = plt.figure(figsize=figsize)
    
    n_rows = 0
    if plot_actual_pwm :
        n_rows += 1
    if plot_opt_pwm :
        n_rows += 1
    if plot_mean_logo and plot_max_logo :
        n_rows *= 2
    
    gs = None
    if plot_percentile :
        gs = gridspec.GridSpec(n_rows, 3, width_ratios=width_ratios, height_ratios=[1 for k in range(n_rows)])
    else :
        gs = gridspec.GridSpec(n_rows, 2, width_ratios=[width_ratios[0], width_ratios[-1]], height_ratios=[1 for k in range(n_rows)])
    
    ax0 = None
    ax1 = None
    ax8 = None
    ax2 = None
    ax3 = None
    ax9 = None
    ax4 = None
    ax5 = None
    ax10 = None
    ax6 = None
    ax7 = None
    ax11 = None
    
    row_i = 0
    logo_col = 2
    if not plot_percentile :
        logo_col = 1
    if plot_mean_logo :
        if plot_actual_pwm :
            ax0 = plt.subplot(gs[row_i, 0])
            ax1 = plt.subplot(gs[row_i, logo_col])
            if plot_percentile :
                ax8 = plt.subplot(gs[row_i, 1])
            row_i += 1
        if plot_opt_pwm :
            ax2 = plt.subplot(gs[row_i, 0])
            ax3 = plt.subplot(gs[row_i, logo_col])
            if plot_percentile :
                ax9 = plt.subplot(gs[row_i, 1])
            row_i += 1
    if plot_max_logo :
        if plot_actual_pwm :
            ax4 = plt.subplot(gs[row_i, 0])
            ax5 = plt.subplot(gs[row_i, logo_col])
            if plot_percentile :
                ax10 = plt.subplot(gs[row_i, 1])
            row_i += 1
        if plot_opt_pwm :
            ax6 = plt.subplot(gs[row_i, 0])
            ax7 = plt.subplot(gs[row_i, logo_col])
            if plot_percentile :
                ax11 = plt.subplot(gs[row_i, 1])
            row_i += 1
    
    stats_ax = [ax0, ax2, ax4, ax6]
    perc_ax = [ax8, ax9, ax10, ax11]
    logo_ax = [ax1, ax3, ax5, ax7]
    
    if plot_mean_logo :
        if plot_actual_pwm :
            plt.sca(stats_ax[0])
            plt.axis('off')
        if plot_opt_pwm :
            plt.sca(stats_ax[1])
            plt.axis('off')
    if plot_max_logo :
        if plot_actual_pwm :
            plt.sca(stats_ax[2])
            plt.axis('off')
        if plot_opt_pwm :
            plt.sca(stats_ax[3])
            plt.axis('off')
    
    if plot_percentile :
        if plot_mean_logo :
            if plot_actual_pwm :
                plt.sca(perc_ax[0])
                plt.axis('off')
            if plot_opt_pwm :
                plt.sca(perc_ax[1])
                plt.axis('off')
        if plot_max_logo :
            if plot_actual_pwm :
                plt.sca(perc_ax[2])
                plt.axis('off')
            if plot_opt_pwm :
                plt.sca(perc_ax[3])
                plt.axis('off')
    
    human_logodds = sorted(np.array(np.ravel(df_human[true_column].values)))
    height_base = (1.0 - logo_height) / 2.
    
    if plot_mean_logo :
        n_samples = len(df_seqs)
        mean_logodds = np.mean(df_seqs[true_column])
        std_logodds = np.std(df_seqs[true_column])
        
        perc = float(len(np.nonzero(human_logodds <= mean_logodds)[0])) / float(len(df_human))
        perc *= 100.

        annot_text = 'Samples = ' + str(int(n_samples))
        #annot_text += '\nLogodds = ' + str(round(mean_logodds, 2)) + ' +- ' + str(round(std_logodds, 2))
        if usage_unit == 'log' :
            annot_text += '\nLogodds = ' + str(round(mean_logodds, 2))
        else :
            usage = 1. / (1. + np.exp(-mean_logodds))
            annot_text += '\nUsage = ' + str(round(usage, 4))
        annot_text += '\nPerc. = ' + str(round(perc, 2)) + '%'
        
        side_plot_i = 0
        if not plot_actual_pwm :
            side_plot_i = 1
        
        stats_ax[side_plot_i].text(0.99, 0.5, annot_text, horizontalalignment='right', verticalalignment='center', transform=stats_ax[side_plot_i].transAxes, color='black', fontsize=12, weight="bold")

        if plot_percentile :
            perc_ax[side_plot_i].plot(np.arange(len(df_human)), human_logodds, linewidth=2, color='black')
            perc_ax[side_plot_i].scatter([len(np.nonzero(human_logodds <= mean_logodds)[0])], [mean_logodds], s=50, c='orange')
            x_coord = len(np.nonzero(human_logodds <= mean_logodds)[0])
            perc_ax[side_plot_i].plot([x_coord, x_coord], [np.min(human_logodds), mean_logodds], color='black', linestyle='--', linewidth=1.5)

        if plot_actual_pwm :
            for j in range(plot_start, plot_end) :
                sort_index = np.argsort(pwm[j, :])

                for ii in range(0, 4) :
                    i = sort_index[ii]

                    nt_prob = pwm[j, i] * conservation[j]

                    nt = ''
                    if i == 0 :
                        nt = 'A'
                    elif i == 1 :
                        nt = 'C'
                    elif i == 2 :
                        nt = 'G'
                    elif i == 3 :
                        nt = 'T'

                    color = None
                    if fixed_seq[j] and black_fixed_seq :
                        color = 'black'

                    if ii == 0 :
                        letterAt(nt, j + 0.5, height_base, nt_prob * logo_height, logo_ax[0], color=color)
                    else :
                        prev_prob = np.sum(pwm[j, sort_index[:ii]] * conservation[j]) * logo_height
                        letterAt(nt, j + 0.5, height_base + prev_prob, nt_prob * logo_height, logo_ax[0], color=color)

            #ax[0].plot([0, pwm.shape[0]], [0, 1], color='black', linestyle='--')

            plt.sca(logo_ax[0])

            plt.xlim((plot_start, plot_end))
            plt.ylim((0, 2))
            plt.xticks([], [])
            plt.yticks([], [])
            plt.axis('off')
            logo_ax[0].axhline(y=0.01 + height_base, color='black', linestyle='-', linewidth=2)


        #Plot optimization pwm

        entropy = np.zeros(opt_pwm.shape)
        entropy[opt_pwm > 0] = opt_pwm[opt_pwm > 0] * -np.log2(opt_pwm[opt_pwm > 0])
        entropy = np.sum(entropy, axis=1)
        conservation = 2 - entropy

        if plot_opt_pwm :
            for j in range(plot_start, plot_end) :
                sort_index = np.argsort(opt_pwm[j, :])

                for ii in range(0, 4) :
                    i = sort_index[ii]

                    nt_prob = opt_pwm[j, i] * conservation[j]

                    nt = ''
                    if i == 0 :
                        nt = 'A'
                    elif i == 1 :
                        nt = 'C'
                    elif i == 2 :
                        nt = 'G'
                    elif i == 3 :
                        nt = 'T'

                    color = None
                    if fixed_seq[j] and black_fixed_seq :
                        color = 'black'

                    if ii == 0 :
                        letterAt(nt, j + 0.5, height_base, nt_prob * logo_height, logo_ax[1], color=color)
                    else :
                        prev_prob = np.sum(opt_pwm[j, sort_index[:ii]] * conservation[j]) * logo_height
                        letterAt(nt, j + 0.5, height_base + prev_prob, nt_prob * logo_height, logo_ax[1], color=color)

            #ax[0].plot([0, pwm.shape[0]], [0, 1], color='black', linestyle='--')

            plt.sca(logo_ax[1])

            plt.xlim((plot_start, plot_end))
            plt.ylim((0, 2))
            plt.xticks([], [])
            plt.yticks([], [])
            plt.axis('off')
            logo_ax[1].axhline(y=0.01 + height_base, color='black', linestyle='-', linewidth=2)
    
    
    if plot_max_logo :
        
        wt_max_sort_index = np.argsort(wt_mean_logodds)[::-1]
        
        wt_max_index = 0
        if max_index == 'mid' :
            wt_max_index = wt_max_sort_index[int(len(wt_max_sort_index) / 2)]
        else :
            wt_max_index = wt_max_sort_index[max_index]
    
        df_seq = df_seqs.query("wt_seq == '" + wt_seqs[wt_max_index] + "'")
        
        seqs = list(df_seq['master_seq'].values)
        n_seqs = len(seqs)
        pwm = get_pwm(seqs)

        wt_seq = wt_seqs[wt_max_index]
        n_samples = len(df_seq)
        wt_mean_logodds = wt_mean_logodds[wt_max_index]
        wt_std_logodds = wt_std_logodds[wt_max_index]
        if override_mean_stats :
            n_samples = len(df_seqs)
            wt_mean_logodds = np.mean(df_seqs[true_column])
            wt_std_logodds = np.std(df_seqs[true_column])

        subexperiment = list(df_seqs.query("wt_seq == '" + wt_seq + "'")['subexperiment'].values)[0]
        opt_pwm = np.vstack([np.ones((1, 4)) * 0.25, find_wt_pwm(wt_seq, max_iso_pwm_dict[gene + '_' + subexperiment])])[:164, :]

        if np.sum(opt_pwm) > 0 :
            for j in range(0, opt_pwm.shape[0]) :
                if np.sum(opt_pwm[j, :]) > 0 :
                    opt_pwm[j, :] /= np.sum(opt_pwm[j, :])

        #Slice according to seq trim index
        seqs = [seq[seq_trim_start: seq_trim_end] for seq in seqs]
        pwm = pwm[:, seq_trim_start: seq_trim_end]
        opt_pwm = opt_pwm[:, seq_trim_start: seq_trim_end]

        pwm += 0.001
        for j in range(0, pwm.shape[0]) :
            pwm[j, :] /= np.sum(pwm[j, :])

        #Plot actual array pwm

        entropy = np.zeros(pwm.shape)
        entropy[pwm > 0] = pwm[pwm > 0] * -np.log2(pwm[pwm > 0])
        entropy = np.sum(entropy, axis=1)
        conservation = 2 - entropy
        
        
        perc = float(len(np.nonzero(human_logodds <= wt_mean_logodds)[0])) / float(len(df_human))
        perc *= 100.

        annot_text = 'Samples = ' + str(int(n_samples))
        #annot_text += '\nLogodds = ' + str(round(wt_mean_logodds, 2)) + ' +- ' + str(round(wt_std_logodds, 2))
        if usage_unit == 'log' :
            annot_text += '\nLogodds = ' + str(round(wt_mean_logodds, 2))
        else :
            usage = 1. / (1. + np.exp(-wt_mean_logodds))
            annot_text += '\nUsage = ' + str(round(usage, 4))
        annot_text += '\nPerc. = ' + str(round(perc, 2)) + '%'
        
        side_plot_i = 2
        if not plot_actual_pwm :
            side_plot_i = 3
        
        stats_ax[side_plot_i].text(0.99, 0.5, annot_text, horizontalalignment='right', verticalalignment='center', transform=stats_ax[side_plot_i].transAxes, color='black', fontsize=12, weight="bold")

        if plot_percentile :
            perc_ax[side_plot_i].plot(np.arange(len(df_human)), human_logodds, linewidth=2, color='black')
            perc_ax[side_plot_i].scatter([len(np.nonzero(human_logodds <= wt_mean_logodds)[0])], [wt_mean_logodds], s=50, c='orange')
            x_coord = len(np.nonzero(human_logodds <= wt_mean_logodds)[0])
            perc_ax[side_plot_i].plot([x_coord, x_coord], [np.min(human_logodds), wt_mean_logodds], color='black', linestyle='--', linewidth=1.5)

        if plot_actual_pwm :
            for j in range(plot_start, plot_end) :
                sort_index = np.argsort(pwm[j, :])

                for ii in range(0, 4) :
                    i = sort_index[ii]

                    nt_prob = pwm[j, i] * conservation[j]

                    nt = ''
                    if i == 0 :
                        nt = 'A'
                    elif i == 1 :
                        nt = 'C'
                    elif i == 2 :
                        nt = 'G'
                    elif i == 3 :
                        nt = 'T'

                    color = None
                    if fixed_seq[j] and black_fixed_seq :
                        color = 'black'

                    if ii == 0 :
                        letterAt(nt, j + 0.5, height_base, nt_prob * logo_height, logo_ax[2], color=color)
                    else :
                        prev_prob = np.sum(pwm[j, sort_index[:ii]] * conservation[j]) * logo_height
                        letterAt(nt, j + 0.5, height_base + prev_prob, nt_prob * logo_height, logo_ax[2], color=color)

            #ax[0].plot([0, pwm.shape[0]], [0, 1], color='black', linestyle='--')

            plt.sca(logo_ax[2])

            plt.xlim((plot_start, plot_end))
            plt.ylim((0, 2))
            plt.xticks([], [])
            plt.yticks([], [])
            plt.axis('off')
            logo_ax[2].axhline(y=0.01 + height_base, color='black', linestyle='-', linewidth=2)


        #Plot optimization pwm

        entropy = np.zeros(opt_pwm.shape)
        entropy[opt_pwm > 0] = opt_pwm[opt_pwm > 0] * -np.log2(opt_pwm[opt_pwm > 0])
        entropy = np.sum(entropy, axis=1)
        conservation = 2 - entropy

        if plot_opt_pwm :
            for j in range(plot_start, plot_end) :
                sort_index = np.argsort(opt_pwm[j, :])

                for ii in range(0, 4) :
                    i = sort_index[ii]

                    nt_prob = opt_pwm[j, i] * conservation[j]

                    nt = ''
                    if i == 0 :
                        nt = 'A'
                    elif i == 1 :
                        nt = 'C'
                    elif i == 2 :
                        nt = 'G'
                    elif i == 3 :
                        nt = 'T'

                    color = None
                    if fixed_seq[j] and black_fixed_seq :
                        color = 'black'

                    if ii == 0 :
                        letterAt(nt, j + 0.5, height_base, nt_prob * logo_height, logo_ax[3], color=color)
                    else :
                        prev_prob = np.sum(opt_pwm[j, sort_index[:ii]] * conservation[j]) * logo_height
                        letterAt(nt, j + 0.5, height_base + prev_prob, nt_prob * logo_height, logo_ax[3], color=color)

            #ax[0].plot([0, pwm.shape[0]], [0, 1], color='black', linestyle='--')

            plt.sca(logo_ax[3])

            plt.xlim((plot_start, plot_end))
            plt.ylim((0, 2))
            plt.xticks([], [])
            plt.yticks([], [])
            plt.axis('off')
            logo_ax[3].axhline(y=0.01 + height_base, color='black', linestyle='-', linewidth=2)
    
    
    for axis in fig.axes :
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
    
    plt.tight_layout()
    
    if save_figs :
        plt.savefig(fig_name + '.png', transparent=True, dpi=fig_dpi)
        plt.savefig(fig_name + '.svg')
        plt.savefig(fig_name + '.eps')
    
    plt.show()

#Max Isoform- optimized sequence PWMs (generated by SeqProp)
def load_max_isoform_pwms() :
    file_path = 'max_isoform_logos/'

    max_iso_pwm_dict = {}

    #Doubledope, Simple, Tomm5 v1

    max_iso_pwm_dict['doubledope_max_score_punish_cruns_softer'] = np.load(file_path + 'apa_array_v1/doubledope_max_class_max_score_punish_cruns_softer_1_images_20_tries_final_pwms.npy')[:5,:,:]
    max_iso_pwm_dict['doubledope_max_score_punish_cruns_harder'] = np.load(file_path + 'apa_array_v1/doubledope_max_class_max_score_punish_cruns_harder_1_images_20_tries_final_pwms.npy')[:10,:,:]
    max_iso_pwm_dict['doubledope_max_score_punish_cruns_aruns'] = np.load(file_path + 'apa_array_v3/doubledope_max_class_max_score_punish_cruns_aruns_1_images_20_tries_final_pwms.npy')[:10,:,:]

    max_iso_pwm_dict['doubledope_target_00'] = np.load(file_path + 'apa_array_v1/doubledope_max_class_target00_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['doubledope_target_025'] = np.load(file_path + 'apa_array_v1/doubledope_max_class_target025_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['doubledope_target_05'] = np.load(file_path + 'apa_array_v1/doubledope_max_class_target05_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['doubledope_target_075'] = np.load(file_path + 'apa_array_v1/doubledope_max_class_target075_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['doubledope_target_10'] = np.load(file_path + 'apa_array_v1/doubledope_max_class_target10_1_images_5_tries_final_pwms.npy')


    max_iso_pwm_dict['simple_max_score_punish_cruns_softer'] = np.load(file_path + 'apa_array_v1/simple_max_class_max_score_punish_cruns_softer_1_images_20_tries_final_pwms.npy')[:5,:,:]
    max_iso_pwm_dict['simple_max_score_punish_cruns_harder'] = np.load(file_path + 'apa_array_v1/simple_max_class_max_score_punish_cruns_harder_1_images_20_tries_final_pwms.npy')[:10,:,:]
    max_iso_pwm_dict['simple_max_score_punish_cruns_aruns'] = np.load(file_path + 'apa_array_v3/simple_max_class_max_score_punish_cruns_aruns_1_images_20_tries_final_pwms.npy')[:10,:,:]

    max_iso_pwm_dict['simple_target_00'] = np.load(file_path + 'apa_array_v1/simple_max_class_target00_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['simple_target_025'] = np.load(file_path + 'apa_array_v1/simple_max_class_target025_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['simple_target_05'] = np.load(file_path + 'apa_array_v1/simple_max_class_target05_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['simple_target_075'] = np.load(file_path + 'apa_array_v1/simple_max_class_target075_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['simple_target_10'] = np.load(file_path + 'apa_array_v1/simple_max_class_target10_1_images_5_tries_final_pwms.npy')


    max_iso_pwm_dict['tomm5_max_score_punish_cruns_softer'] = np.load(file_path + 'apa_array_v1/tomm5_max_class_max_score_punish_cruns_softer_1_images_20_tries_final_pwms.npy')[:5,:,:]
    max_iso_pwm_dict['tomm5_max_score_punish_cruns_harder'] = np.load(file_path + 'apa_array_v1/tomm5_max_class_max_score_punish_cruns_harder_1_images_20_tries_final_pwms.npy')[:10,:,:]

    max_iso_pwm_dict['tomm5_target_00'] = np.load(file_path + 'apa_array_v1/tomm5_max_class_target00_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['tomm5_target_025'] = np.load(file_path + 'apa_array_v1/tomm5_max_class_target025_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['tomm5_target_05'] = np.load(file_path + 'apa_array_v1/tomm5_max_class_target05_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['tomm5_target_075'] = np.load(file_path + 'apa_array_v1/tomm5_max_class_target075_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['tomm5_target_10'] = np.load(file_path + 'apa_array_v1/tomm5_max_class_target10_1_images_5_tries_final_pwms.npy')


    #APASIX v1

    max_iso_pwm_dict['aar_max_score_punish_cruns_softer'] = np.load(file_path + 'apa_array_v1/aar_max_class_max_score_punish_cruns_softer_1_images_10_tries_final_pwms.npy')[:5,:,:]
    max_iso_pwm_dict['aar_max_score_punish_cruns_harder'] = np.load(file_path + 'apa_array_v1/aar_max_class_max_score_punish_cruns_harder_1_images_10_tries_final_pwms.npy')[:5,:,:]

    max_iso_pwm_dict['aar_target_00'] = np.load(file_path + 'apa_array_v1/aar_max_class_target00_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['aar_target_025'] = np.load(file_path + 'apa_array_v1/aar_max_class_target025_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['aar_target_05'] = np.load(file_path + 'apa_array_v1/aar_max_class_target05_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['aar_target_075'] = np.load(file_path + 'apa_array_v1/aar_max_class_target075_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['aar_target_10'] = np.load(file_path + 'apa_array_v1/aar_max_class_target10_1_images_5_tries_final_pwms.npy')


    max_iso_pwm_dict['atr_max_score_punish_cruns_softer'] = np.load(file_path + 'apa_array_v1/atr_max_class_max_score_punish_cruns_softer_1_images_10_tries_final_pwms.npy')[:5,:,:]
    max_iso_pwm_dict['atr_max_score_punish_cruns_harder'] = np.load(file_path + 'apa_array_v1/atr_max_class_max_score_punish_cruns_harder_1_images_10_tries_final_pwms.npy')[:5,:,:]

    max_iso_pwm_dict['atr_target_00'] = np.load(file_path + 'apa_array_v1/atr_max_class_target00_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['atr_target_025'] = np.load(file_path + 'apa_array_v1/atr_max_class_target025_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['atr_target_05'] = np.load(file_path + 'apa_array_v1/atr_max_class_target05_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['atr_target_075'] = np.load(file_path + 'apa_array_v1/atr_max_class_target075_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['atr_target_10'] = np.load(file_path + 'apa_array_v1/atr_max_class_target10_1_images_5_tries_final_pwms.npy')


    max_iso_pwm_dict['hsp_max_score_punish_cruns_softer'] = np.load(file_path + 'apa_array_v1/hsp_max_class_max_score_punish_cruns_softer_1_images_10_tries_final_pwms.npy')[:5,:,:]
    max_iso_pwm_dict['hsp_max_score_punish_cruns_harder'] = np.load(file_path + 'apa_array_v1/hsp_max_class_max_score_punish_cruns_harder_1_images_10_tries_final_pwms.npy')[:5,:,:]


    max_iso_pwm_dict['snh_max_score_punish_cruns_softer'] = np.load(file_path + 'apa_array_v1/snh_max_class_max_score_punish_cruns_softer_1_images_10_tries_final_pwms.npy')[:5,:,:]
    max_iso_pwm_dict['snh_max_score_punish_cruns_harder'] = np.load(file_path + 'apa_array_v1/snh_max_class_max_score_punish_cruns_harder_1_images_10_tries_final_pwms.npy')[:5,:,:]


    max_iso_pwm_dict['sox_max_score_punish_cruns_softer'] = np.load(file_path + 'apa_array_v1/sox_max_class_max_score_punish_cruns_softer_1_images_10_tries_final_pwms.npy')[:5,:,:]
    max_iso_pwm_dict['sox_max_score_punish_cruns_harder'] = np.load(file_path + 'apa_array_v1/sox_max_class_max_score_punish_cruns_harder_1_images_10_tries_final_pwms.npy')[:5,:,:]

    max_iso_pwm_dict['sox_target_00'] = np.load(file_path + 'apa_array_v1/sox_max_class_target00_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['sox_target_025'] = np.load(file_path + 'apa_array_v1/sox_max_class_target025_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['sox_target_05'] = np.load(file_path + 'apa_array_v1/sox_max_class_target05_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['sox_target_075'] = np.load(file_path + 'apa_array_v1/sox_max_class_target075_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['sox_target_10'] = np.load(file_path + 'apa_array_v1/sox_max_class_target10_1_images_5_tries_final_pwms.npy')


    max_iso_pwm_dict['wha_max_score_punish_cruns_softer'] = np.load(file_path + 'apa_array_v1/wha_max_class_max_score_punish_cruns_softer_1_images_10_tries_final_pwms.npy')[:5,:,:]
    max_iso_pwm_dict['wha_max_score_punish_cruns_harder'] = np.load(file_path + 'apa_array_v1/wha_max_class_max_score_punish_cruns_harder_1_images_10_tries_final_pwms.npy')[:5,:,:]


    #Doubledope, Simple, Tomm5 v2 low entropy

    max_iso_pwm_dict['doubledope_max_score_punish_cruns_softer_v2'] = np.load(file_path + 'apa_array_v2/doubledope_max_class_max_score_punish_cruns_softer_1_images_20_tries_final_pwms.npy')[:5,:,:]
    max_iso_pwm_dict['doubledope_max_score_punish_cruns_harder_v2'] = np.load(file_path + 'apa_array_v2/doubledope_max_class_max_score_punish_cruns_harder_1_images_20_tries_final_pwms.npy')[:10,:,:]
    max_iso_pwm_dict['doubledope_max_score_punish_cruns_aruns_v2'] = np.load(file_path + 'apa_array_v2/doubledope_max_class_max_score_punish_cruns_aruns_1_images_20_tries_final_pwms.npy')[:10,:,:]
    max_iso_pwm_dict['doubledope_max_score_punish_cruns_aruns_cstf_v2'] = np.load(file_path + 'apa_array_v2/doubledope_max_class_max_score_punish_cruns_aruns_cstf_1_images_20_tries_final_pwms.npy')[:5,:,:]

    max_iso_pwm_dict['doubledope_target_00_v2'] = np.load(file_path + 'apa_array_v2/doubledope_max_class_target00_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['doubledope_target_025_v2'] = np.load(file_path + 'apa_array_v2/doubledope_max_class_target025_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['doubledope_target_05_v2'] = np.load(file_path + 'apa_array_v2/doubledope_max_class_target05_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['doubledope_target_075_v2'] = np.load(file_path + 'apa_array_v2/doubledope_max_class_target075_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['doubledope_target_10_v2'] = np.load(file_path + 'apa_array_v2/doubledope_max_class_target10_1_images_5_tries_final_pwms.npy')


    max_iso_pwm_dict['simple_max_score_punish_cruns_softer_v2'] = np.load(file_path + 'apa_array_v2/simple_max_class_max_score_punish_cruns_softer_1_images_20_tries_final_pwms.npy')[:5,:,:]
    max_iso_pwm_dict['simple_max_score_punish_cruns_harder_v2'] = np.load(file_path + 'apa_array_v2/simple_max_class_max_score_punish_cruns_harder_1_images_20_tries_final_pwms.npy')[:10,:,:]
    max_iso_pwm_dict['simple_max_score_punish_cruns_aruns_v2'] = np.load(file_path + 'apa_array_v2/simple_max_class_max_score_punish_cruns_aruns_1_images_20_tries_final_pwms.npy')[:10,:,:]
    max_iso_pwm_dict['simple_max_score_punish_cruns_aruns_cstf_v2'] = np.load(file_path + 'apa_array_v2/simple_max_class_max_score_punish_cruns_aruns_cstf_1_images_20_tries_final_pwms.npy')[:5,:,:]

    max_iso_pwm_dict['simple_target_00_v2'] = np.load(file_path + 'apa_array_v2/simple_max_class_target00_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['simple_target_025_v2'] = np.load(file_path + 'apa_array_v2/simple_max_class_target025_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['simple_target_05_v2'] = np.load(file_path + 'apa_array_v2/simple_max_class_target05_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['simple_target_075_v2'] = np.load(file_path + 'apa_array_v2/simple_max_class_target075_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['simple_target_10_v2'] = np.load(file_path + 'apa_array_v2/simple_max_class_target10_1_images_5_tries_final_pwms.npy')


    max_iso_pwm_dict['tomm5_max_score_punish_cruns_softer_v2'] = np.load(file_path + 'apa_array_v2/tomm5_max_class_max_score_punish_cruns_softer_1_images_20_tries_final_pwms.npy')[:5,:,:]
    max_iso_pwm_dict['tomm5_max_score_punish_cruns_harder_v2'] = np.load(file_path + 'apa_array_v2/tomm5_max_class_max_score_punish_cruns_harder_1_images_20_tries_final_pwms.npy')[:10,:,:]

    max_iso_pwm_dict['tomm5_target_00_v2'] = np.load(file_path + 'apa_array_v2/tomm5_max_class_target00_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['tomm5_target_025_v2'] = np.load(file_path + 'apa_array_v2/tomm5_max_class_target025_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['tomm5_target_05_v2'] = np.load(file_path + 'apa_array_v2/tomm5_max_class_target05_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['tomm5_target_075_v2'] = np.load(file_path + 'apa_array_v2/tomm5_max_class_target075_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['tomm5_target_10_v2'] = np.load(file_path + 'apa_array_v2/tomm5_max_class_target10_1_images_5_tries_final_pwms.npy')


    #APASIX v2 low entropy

    max_iso_pwm_dict['aar_max_score_punish_cruns_softer_v2'] = np.load(file_path + 'apa_array_v2/aar_max_class_max_score_punish_cruns_softer_1_images_10_tries_final_pwms.npy')[:5,:,:]
    max_iso_pwm_dict['aar_max_score_punish_cruns_harder_v2'] = np.load(file_path + 'apa_array_v2/aar_max_class_max_score_punish_cruns_harder_1_images_10_tries_final_pwms.npy')[:5,:,:]

    max_iso_pwm_dict['aar_target_00_v2'] = np.load(file_path + 'apa_array_v2/aar_max_class_target00_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['aar_target_025_v2'] = np.load(file_path + 'apa_array_v2/aar_max_class_target025_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['aar_target_05_v2'] = np.load(file_path + 'apa_array_v2/aar_max_class_target05_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['aar_target_075_v2'] = np.load(file_path + 'apa_array_v2/aar_max_class_target075_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['aar_target_10_v2'] = np.load(file_path + 'apa_array_v2/aar_max_class_target10_1_images_5_tries_final_pwms.npy')


    max_iso_pwm_dict['atr_max_score_punish_cruns_softer_v2'] = np.load(file_path + 'apa_array_v2/atr_max_class_max_score_punish_cruns_softer_1_images_10_tries_final_pwms.npy')[:5,:,:]
    max_iso_pwm_dict['atr_max_score_punish_cruns_harder_v2'] = np.load(file_path + 'apa_array_v2/atr_max_class_max_score_punish_cruns_harder_1_images_10_tries_final_pwms.npy')[:5,:,:]

    max_iso_pwm_dict['atr_target_00_v2'] = np.load(file_path + 'apa_array_v2/atr_max_class_target00_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['atr_target_025_v2'] = np.load(file_path + 'apa_array_v2/atr_max_class_target025_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['atr_target_05_v2'] = np.load(file_path + 'apa_array_v2/atr_max_class_target05_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['atr_target_075_v2'] = np.load(file_path + 'apa_array_v2/atr_max_class_target075_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['atr_target_10_v2'] = np.load(file_path + 'apa_array_v2/atr_max_class_target10_1_images_5_tries_final_pwms.npy')


    max_iso_pwm_dict['hsp_max_score_punish_cruns_softer_v2'] = np.load(file_path + 'apa_array_v2/hsp_max_class_max_score_punish_cruns_softer_1_images_10_tries_final_pwms.npy')[:5,:,:]
    max_iso_pwm_dict['hsp_max_score_punish_cruns_harder_v2'] = np.load(file_path + 'apa_array_v2/hsp_max_class_max_score_punish_cruns_harder_1_images_10_tries_final_pwms.npy')[:5,:,:]


    max_iso_pwm_dict['snh_max_score_punish_cruns_softer_v2'] = np.load(file_path + 'apa_array_v2/snh_max_class_max_score_punish_cruns_softer_1_images_10_tries_final_pwms.npy')[:5,:,:]
    max_iso_pwm_dict['snh_max_score_punish_cruns_harder_v2'] = np.load(file_path + 'apa_array_v2/snh_max_class_max_score_punish_cruns_harder_1_images_10_tries_final_pwms.npy')[:5,:,:]


    max_iso_pwm_dict['sox_max_score_punish_cruns_softer_v2'] = np.load(file_path + 'apa_array_v2/sox_max_class_max_score_punish_cruns_softer_1_images_10_tries_final_pwms.npy')[:5,:,:]
    max_iso_pwm_dict['sox_max_score_punish_cruns_harder_v2'] = np.load(file_path + 'apa_array_v2/sox_max_class_max_score_punish_cruns_harder_1_images_10_tries_final_pwms.npy')[:5,:,:]

    max_iso_pwm_dict['sox_target_00_v2'] = np.load(file_path + 'apa_array_v2/sox_max_class_target00_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['sox_target_025_v2'] = np.load(file_path + 'apa_array_v2/sox_max_class_target025_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['sox_target_05_v2'] = np.load(file_path + 'apa_array_v2/sox_max_class_target05_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['sox_target_075_v2'] = np.load(file_path + 'apa_array_v2/sox_max_class_target075_1_images_5_tries_final_pwms.npy')
    max_iso_pwm_dict['sox_target_10_v2'] = np.load(file_path + 'apa_array_v2/sox_max_class_target10_1_images_5_tries_final_pwms.npy')


    max_iso_pwm_dict['wha_max_score_punish_cruns_softer_v2'] = np.load(file_path + 'apa_array_v2/wha_max_class_max_score_punish_cruns_softer_1_images_10_tries_final_pwms.npy')[:5,:,:]
    max_iso_pwm_dict['wha_max_score_punish_cruns_harder_v2'] = np.load(file_path + 'apa_array_v2/wha_max_class_max_score_punish_cruns_harder_1_images_10_tries_final_pwms.npy')[:5,:,:]

    return max_iso_pwm_dict


