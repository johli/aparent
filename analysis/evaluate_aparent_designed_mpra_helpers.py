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


#Max Cut Helper Functions

def plot_cut_profile(cut_df_filtered, cut_to_experiment, cut_poses, objective_poses, human_cutprob, figsize=(8, 5.5), save_fig_name=None, fig_dpi=150, plot_mode='mean', n_samples=None) :
    f = plt.figure(figsize=figsize)

    ls = []

    for cut_pos in cut_to_experiment :

        keep_index = np.nonzero(cut_df_filtered['subexperiment'].isin(cut_to_experiment[cut_pos]))[0]
        prox_prob = np.array(cut_df_filtered.iloc[keep_index]['proxcut_prob_true'].values.tolist())
        
        if plot_mode == 'mean' :
            if n_samples is not None :
                shuffle_index = np.arange(prox_prob.shape[0])
                np.random.shuffle(shuffle_index)
                prox_prob = prox_prob[shuffle_index[:n_samples], :]
            
            prox_prob = prox_prob.mean(axis=0)
            prox_prob = np.ravel(prox_prob)
        elif plot_mode == 'max' :
            losses = np.array(cut_df_filtered.iloc[keep_index]['loss_logloss'].values)
            sort_index = np.argsort(losses)
            prox_prob = prox_prob[sort_index, :]

            prox_prob = prox_prob[:n_samples, :]
            if n_samples > 1 :
                prox_prob = prox_prob.mean(axis=0)
            prox_prob = np.ravel(prox_prob)
        
        l1, = plt.plot(cut_poses, prox_prob, linewidth=2, label='Objective ' + str(cut_pos))

        fill_x_coords = np.concatenate([np.array([np.min(cut_poses)]), cut_poses, np.array([np.max(cut_poses)])], axis=0)
        fill_y_coords = np.concatenate([np.array([0]), prox_prob, np.array([0])], axis=0)
        plt.fill(fill_x_coords, fill_y_coords, alpha=0.2)

        ls.append(l1)

    la = plt.axvline(x=np.argmax(human_cutprob) + cut_poses[0] + 1, linewidth=2, linestyle='--', color='black', alpha=0.7, label='Native')
    ls.append(la)

    plt.legend(handles=ls, fontsize=14, loc='upper right', bbox_to_anchor=(1.05, 1.3), frameon=True, framealpha=1.0)
    plt.xticks(objective_poses, objective_poses, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(np.min(cut_poses), np.max(cut_poses))
    plt.ylim(0)

    plt.xlabel('Cleavage position from end of CSE', fontsize=18)
    plt.ylabel('Cleavage proportion', fontsize=18)

    plt.tight_layout()
    if save_fig_name is not None :
        plt.savefig(save_fig_name + '.png', transparent=True, dpi=fig_dpi)
        plt.savefig(save_fig_name + '.svg')
        plt.savefig(save_fig_name + '.eps')
    
    plt.show()

def plot_cut_map(cut_df_filtered, cut_to_experiment, cut_poses, objective_poses, human_cutprob, figsize=(4, 6), save_fig_name=None, fig_dpi=150, plot_mode='mean', n_samples=None) :
    cut_probs = []
    cut_label_coords = []
    prev_label_coords = [0]
    cut_labels = []

    f = plt.figure(figsize=figsize)

    for objective_pos, _ in sorted(cut_to_experiment.items(), key=lambda kv: kv[0]) :

        keep_index = np.nonzero(cut_df_filtered['subexperiment'].isin(cut_to_experiment[objective_pos]))[0]
        prox_prob = np.array(cut_df_filtered.iloc[keep_index]['proxcut_prob_true'].values.tolist())

        if plot_mode == 'mean' :
            shuffle_index = np.arange(prox_prob.shape[0])
            np.random.shuffle(shuffle_index)
            if n_samples is not None :
                prox_prob = prox_prob[shuffle_index[:n_samples], :]
        elif plot_mode == 'max' :
            losses = np.array(cut_df_filtered.iloc[keep_index]['loss_logloss'].values)
            sort_index = np.argsort(losses)
            prox_prob = prox_prob[sort_index, :]
            prox_prob = prox_prob[:n_samples, :]
            
            shuffle_index = np.arange(prox_prob.shape[0])
            np.random.shuffle(shuffle_index)
            prox_prob = prox_prob[shuffle_index, :]

        cut_probs.append(prox_prob)

        cut_labels.append(str(objective_pos))

        cut_label_coords.append(prev_label_coords[-1] + float(prox_prob.shape[0]) / 2.)
        prev_label_coords.append(prev_label_coords[-1] + float(prox_prob.shape[0]))

        plt.axhline(y=prev_label_coords[-1], color='black', linewidth=2, linestyle='--')
        plt.axvline(x=objective_pos, color='orange', linewidth=2, linestyle='--', alpha=0.2)

    cut_probs = np.vstack(cut_probs)

    plt.imshow(np.concatenate([np.zeros((cut_probs.shape[0], cut_poses[0])), cut_probs], axis=1), cmap='Greens', vmin=0.05, vmax=0.3, aspect='auto')

    plt.xlabel('Cleavage position', fontsize=18)
    plt.ylabel('Cleavage objective', fontsize=18)

    plt.xlim(1, cut_probs.shape[1] + 1)

    ax = plt.gca()
    ax.set_xticks(objective_poses)
    ax.set_xticklabels(objective_poses, fontsize=14, ha='center', va='top')

    ax.set_yticks(cut_label_coords)
    ax.set_yticklabels(cut_labels, fontsize=14, ha='right', va='center')

    plt.tight_layout()
    if save_fig_name is not None :
        plt.savefig(save_fig_name + '.png', transparent=True, dpi=fig_dpi)
        plt.savefig(save_fig_name + '.svg')
        plt.savefig(save_fig_name + '.eps')
    
    plt.show()

def plot_position_scatter(cut_df_filtered, cut_to_experiment, cut_poses, objective_poses, human_cutprob, variant_filter="variant == 'wt'", figsize=(5, 5), save_fig_name=None, fig_dpi=150) :
    f = plt.figure(figsize=figsize)

    avgpos_true_all = []
    avgpos_pred_all = []

    for cut_pos, _ in sorted(cut_to_experiment.items(), key=lambda kv: kv[0]) :

        keep_index = np.nonzero(cut_df_filtered['subexperiment'].isin(cut_to_experiment[cut_pos]))[0]
        avgpos_true = np.ravel(np.array(cut_df_filtered.iloc[keep_index].query(variant_filter)['avgpos_true']))
        avgpos_pred = np.ravel(np.array(cut_df_filtered.iloc[keep_index].query(variant_filter)['avgpos_pred']))

        avgpos_true_all.append(avgpos_true)
        avgpos_pred_all.append(avgpos_pred)
        
        plt.scatter(avgpos_pred - (56 + cut_poses[0]), avgpos_true - (56 + cut_poses[0]), s=15, label=str(cut_pos), alpha=0.5)

    avgpos_true_all = np.ravel(np.concatenate(avgpos_true_all, axis=0))
    avgpos_pred_all = np.ravel(np.concatenate(avgpos_pred_all, axis=0))
    rval, _ = pearsonr(avgpos_true_all, avgpos_pred_all)
    
    plt.xlabel('Predicted cut position', fontsize=16)
    plt.ylabel('Observed cut position', fontsize=16)
    plt.xticks(objective_poses, objective_poses, fontsize=12)
    plt.yticks(objective_poses, objective_poses, fontsize=12)

    plt.title('R^2 = ' + str(round(rval * rval, 2)), fontsize=16)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    if save_fig_name is not None :
        plt.savefig(save_fig_name + '.png', transparent=True, dpi=fig_dpi)
        plt.savefig(save_fig_name + '.eps')
    
    plt.show()

def plot_cut_logo(df, df_human, max_cut_pwm_dict, max_cut_yhat_dict, gene, subexperiments, plot_mfe=False, fold_height=0.5, annotate_peaks=False, normalize_probs=False, normalize_range=[57, 105], agg_mode='avg', override_mean_stats=False, plot_percentile=True, plot_mean_logo=True, plot_max_logo=True, plot_actual_pwm=True, plot_opt_pwm=True, black_fixed_seq=True, max_index=None, true_column='median_proximal_vs_distal_logodds_true', cut_column='pooled_cut_prob_true', figsize=(12, 3), width_ratios=[1, 7], logo_height=1.0, usage_unit='log', plot_snvs=False, seq_trim_start=0, seq_trim_end=164, plot_start=0, plot_end=164, pas_downscaling=1.0, save_figs=False, fig_name=None, fig_dpi=300) :

    #Make sequence logo
    
    df_seqs = df.copy()#df.query("variant == 'wt' or variant == 'sampled'")
    df_seqs = df_seqs.loc[df_seqs.subexperiment.isin(subexperiments)]
    
    #Mean logos
    seqs = list(df_seqs['master_seq'].values)
    
    n_seqs = len(seqs)
    pwm = get_pwm(seqs)
    
    prob = np.zeros(pwm.shape[0])
    for cut_prob in list(df_seqs[cut_column].values) :
        if normalize_probs :
            if normalize_range is not None :
                prob[normalize_range[0]:normalize_range[1]] += cut_prob[normalize_range[0]:normalize_range[1]] / np.sum(cut_prob[normalize_range[0]:normalize_range[1]])
            else :
                prob += cut_prob[:164] / np.sum(cut_prob[:164])
        else :
            prob += cut_prob[:164]
    
    if agg_mode in ['avg', 'max'] :
        prob /= float(len(seqs))
    elif agg_mode in ['pool', 'max'] :
        prob /= np.sum(prob)
    
    pred_prob = np.zeros(pwm.shape[0])
    for cut_prob in list(df_seqs['cut_prob_pred'].values) :
        if normalize_probs :
            if normalize_range is not None :
                pred_prob[normalize_range[0]:normalize_range[1]] += cut_prob[normalize_range[0]:normalize_range[1]] / np.sum(cut_prob[normalize_range[0]:normalize_range[1]])
            else :
                pred_prob += cut_prob[:164] / np.sum(cut_prob[:164])
        else :
            pred_prob += cut_prob[:164]
    
    if agg_mode in ['avg', 'max'] :
        pred_prob /= float(len(seqs))
    elif agg_mode in ['pool', 'max'] :
        pred_prob /= np.sum(pred_prob)
    
    wt_seqs = list(df_seqs['wt_seq'].unique())
    
    wt_mean_logodds = np.zeros(len(wt_seqs))
    wt_avgpos = np.zeros(len(wt_seqs))
    wt_logloss = np.zeros(len(wt_seqs))
    
    #Get wt seq cluster statistics
    for i, wt_seq in enumerate(wt_seqs) :
        #wt_mean_logodds[i] = np.mean(df_seqs.query("wt_seq == '" + wt_seq + "'")[true_column])
        wt_usage_list = [np.sum(cut_prob[57:105]) for cut_prob in list(df_seqs.query("wt_seq == '" + wt_seq + "'")[cut_column].values)]
        if normalize_probs and normalize_range is not None :
            wt_usage_list = [np.sum(cut_prob[normalize_range[0]:normalize_range[1]]) / (np.sum(cut_prob[normalize_range[0]:normalize_range[1]]) + cut_prob[-1]) for cut_prob in list(df_seqs.query("wt_seq == '" + wt_seq + "'")[cut_column].values)]
        wt_logodds_list = [np.log(p / (1. - p)) for p in wt_usage_list]
        wt_mean_logodds[i] = np.mean(np.array(wt_logodds_list))
        
        wt_avgpos[i] = np.mean(df_seqs.query("wt_seq == '" + wt_seq + "'")['avgpos_true'])
        if agg_mode == 'max' :
            wt_logloss[i] = np.min(df_seqs.query("wt_seq == '" + wt_seq + "'")['loss_logloss'])
        else :
            wt_logloss[i] = np.mean(df_seqs.query("wt_seq == '" + wt_seq + "'")['loss_logloss'])
    
    opt_pwm = np.zeros(pwm.shape)
    opt_prob = np.zeros(pwm.shape[0])
    n_opt = 0
    for wt_seq in wt_seqs :
        subexperiment = list(df_seqs.query("wt_seq == '" + wt_seq + "'")['subexperiment'].values)[0]
        subexp_pwm = np.vstack([np.ones((1, 4)) * 0.25, find_wt_pwm(wt_seq, max_cut_pwm_dict[gene + '_' + subexperiment])])[:164, :]
        subexp_prob = find_wt_yhat(wt_seq, max_cut_pwm_dict[gene + '_' + subexperiment], max_cut_yhat_dict[gene + '_' + subexperiment])
        
        opt_pwm += subexp_pwm
        
        if normalize_probs :
            if normalize_range is not None :
                opt_prob[normalize_range[0]:normalize_range[1]] += subexp_prob[normalize_range[0]-1:normalize_range[1]-1] / np.sum(subexp_prob[normalize_range[0]-1:normalize_range[1]-1])
            else :
                opt_prob[1:] += subexp_prob[:163] / np.sum(subexp_prob[:163])
        else :
            opt_prob[1:] += subexp_prob[:163]
        n_opt += 1.
    
    
    fixed_seq = []
    if np.sum(opt_pwm) > 0 :
        if agg_mode in ['avg', 'max'] :
            opt_prob /= n_opt
        elif agg_mode in ['pool', 'max'] :
            opt_prob /= np.sum(opt_prob)
        
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
    opt_prob = opt_prob[seq_trim_start: seq_trim_end]
    
    pwm += 0.001
    for j in range(0, pwm.shape[0]) :
        pwm[j, :] /= np.sum(pwm[j, :])
    
    entropy = np.zeros(pwm.shape)
    entropy[pwm > 0] = pwm[pwm > 0] * -np.log2(pwm[pwm > 0])
    entropy = np.sum(entropy, axis=1)
    conservation = 2 - entropy

    fig = plt.figure(figsize=figsize)
    
    n_rows = 0
    if plot_actual_pwm :
        n_rows += 2
    if plot_opt_pwm :
        n_rows += 2
    if plot_mean_logo and plot_max_logo :
        n_rows *= 2
    
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
    
    #cut axes
    ax12 = None
    ax13 = None
    ax14 = None
    ax15 = None
    
    row_i = 1
    if plot_mean_logo :
        if plot_actual_pwm :
            ax0 = plt.subplot(gs[row_i, 0])
            ax1 = plt.subplot(gs[row_i, 1])
            if plot_percentile :
                ax8 = plt.subplot(gs[row_i-1, 0])
            ax12 = plt.subplot(gs[row_i-1, 1])
            row_i += 2
        if plot_opt_pwm :
            ax2 = plt.subplot(gs[row_i, 0])
            ax3 = plt.subplot(gs[row_i, 1])
            if plot_percentile :
                ax9 = plt.subplot(gs[row_i-1, 0])
            ax13 = plt.subplot(gs[row_i-1, 1])
            row_i += 2
    if plot_max_logo :
        if plot_actual_pwm :
            ax4 = plt.subplot(gs[row_i, 0])
            ax5 = plt.subplot(gs[row_i, 1])
            if plot_percentile :
                ax10 = plt.subplot(gs[row_i-1, 0])
            ax14 = plt.subplot(gs[row_i-1, 1])
            row_i += 2
        if plot_opt_pwm :
            ax6 = plt.subplot(gs[row_i, 0])
            ax7 = plt.subplot(gs[row_i, 1])
            if plot_percentile :
                ax11 = plt.subplot(gs[row_i-1, 0])
            ax15 = plt.subplot(gs[row_i-1, 1])
            row_i += 2
    
    stats_ax = [ax0, ax2, ax4, ax6]
    perc_ax = [ax8, ax9, ax10, ax11]
    logo_ax = [ax1, ax3, ax5, ax7]
    
    cut_ax = [ax12, ax13, ax14, ax15]
    
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
    
    human_logodds = sorted(np.array(np.ravel(df_human[true_column[:-5]].values)))
    height_base = (1.0 - logo_height) / 2.
    
    objective_pos = int(subexperiments[0].split('_')[-1]) - 49
    human_cutprob = np.mean(np.array(df_human[cut_column].values.tolist())[:, 50:110], axis=0)
    
    if plot_mean_logo :
        n_samples = len(df_seqs)
        mean_logodds = np.mean(wt_mean_logodds)#np.mean(df_seqs[true_column])
        std_logodds = np.std(wt_mean_logodds)
        
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
            perc_ax[side_plot_i].plot(np.arange(len(human_cutprob)), human_cutprob, linewidth=2, color='black')
            perc_ax[side_plot_i].scatter([objective_pos], [human_cutprob[objective_pos]], s=50, c='red')
            
            if objective_pos <= 30 :
                perc_ax[side_plot_i].annotate('Objective', xy=(objective_pos, human_cutprob[objective_pos]), xycoords='data', xytext=(0.55, 0.8), fontsize=10, weight="bold", color='red', textcoords='axes fraction', arrowprops=dict(connectionstyle="arc3,rad=-.2", headlength=8, headwidth=8, shrink=0.05, width=1.5, color='black'))
            else :
                perc_ax[side_plot_i].annotate('Objective', xy=(objective_pos, human_cutprob[objective_pos]), xycoords='data', xytext=(0.55, 0.8), fontsize=10, weight="bold", color='red', textcoords='axes fraction', arrowprops=dict(connectionstyle="arc3,rad=.2", headlength=8, headwidth=8, shrink=0.05, width=1.5, color='black'))

        if plot_actual_pwm :
            
            l2, = cut_ax[0].plot(np.arange(plot_end - plot_start) + plot_start, prob[plot_start:plot_end], linewidth=3, linestyle='-', label='Observed', color='black', alpha=0.7)
            l1, = cut_ax[0].plot(np.arange(plot_end - plot_start) + plot_start, pred_prob[plot_start:plot_end], linewidth=3, linestyle='-', label='Predicted', color='red', alpha=0.7)
            
            if annotate_peaks :
                annot_text = str(int(round(prob[objective_pos + 50] * 100, 0))) + '% Cleavage'
                cut_ax[2].annotate(annot_text, xy=(objective_pos + 50, prob[objective_pos + 50]), xycoords='data', xytext=(-30, -5), ha='right', fontsize=10, weight="bold", color='black', textcoords='offset points', arrowprops=dict(connectionstyle="arc3,rad=-.1", headlength=8, headwidth=8, shrink=0.15, width=1.5, color='black'))
            
            plt.sca(cut_ax[0])

            plt.xlim((plot_start, plot_end))
            #plt.ylim((0, 2))
            plt.xticks([], [])
            plt.yticks([], [])
            plt.legend(handles=[l1, l2], fontsize=12, prop=dict(weight='bold'), frameon=False)
            plt.axis('off')
            
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
            
            l2, = cut_ax[1].plot(np.arange(plot_end - plot_start) + plot_start, prob[plot_start:plot_end], linewidth=3, linestyle='-', label='Observed', color='black', alpha=0.7)
            l1, = cut_ax[1].plot(np.arange(plot_end - plot_start) + plot_start, opt_prob[plot_start:plot_end], linewidth=3, linestyle='-', label='Predicted', color='red', alpha=0.7)
            
            if annotate_peaks :
                annot_text = str(int(round(prob[objective_pos + 50] * 100, 0))) + '% Cleavage'
                cut_ax[2].annotate(annot_text, xy=(objective_pos + 50, prob[objective_pos + 50]), xycoords='data', xytext=(-30, -5), ha='right', fontsize=10, weight="bold", color='black', textcoords='offset points', arrowprops=dict(connectionstyle="arc3,rad=-.1", headlength=8, headwidth=8, shrink=0.15, width=1.5, color='black'))
            
            plt.sca(cut_ax[1])

            plt.xlim((plot_start, plot_end))
            #plt.ylim((0, 2))
            plt.xticks([], [])
            plt.yticks([], [])
            plt.legend(handles=[l1, l2], fontsize=12, prop=dict(weight='bold'), frameon=False)
            plt.axis('off')
            
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
        
        wt_max_sort_index = np.argsort(wt_logloss)
        
        wt_max_index = 0
        if max_index == 'mid' :
            wt_max_index = wt_max_sort_index[int(len(wt_max_sort_index) / 2)]
        else :
            wt_max_index = wt_max_sort_index[max_index]
    
        df_seq = df_seqs.query("wt_seq == '" + wt_seqs[wt_max_index] + "'")
        
        seqs = list(df_seq['master_seq'].values)
        n_seqs = len(seqs)
        pwm = get_pwm(seqs)
        
        prob = np.zeros(pwm.shape[0])
        pred_prob = np.zeros(pwm.shape[0])
        if agg_mode == 'max' :
            in_logloss = np.zeros(len(seqs))

            #Get wt seq cluster statistics
            for i, seq in enumerate(seqs) :
                in_logloss[i] = list(df_seq.query("master_seq == '" + seq + "'")['loss_logloss'].values)[0]
            
            in_max_sort_index = np.argsort(in_logloss)
            
            max_seq = seqs[in_max_sort_index[0]]
            cut_prob = list(df_seq.query("master_seq == '" + max_seq + "'")[cut_column].values)[0]
            if normalize_probs :
                if normalize_range is not None :
                    prob[normalize_range[0]:normalize_range[1]] = cut_prob[normalize_range[0]:normalize_range[1]] / np.sum(cut_prob[normalize_range[0]:normalize_range[1]])
                else :
                    prob = cut_prob[:164] / np.sum(cut_prob[:164])
            else :
                prob = cut_prob[:164]
            
            cut_prob = list(df_seq.query("master_seq == '" + max_seq + "'")['cut_prob_pred'].values)[0]
            if normalize_probs :
                if normalize_range is not None :
                    pred_prob[normalize_range[0]:normalize_range[1]] = cut_prob[normalize_range[0]:normalize_range[1]] / np.sum(cut_prob[normalize_range[0]:normalize_range[1]])
                else :
                    pred_prob = cut_prob[:164] / np.sum(cut_prob[:164])
            else :
                pred_prob = cut_prob[:164]
        else :
            for seq_i, cut_prob in enumerate(list(df_seq[cut_column].values)) :
                if normalize_probs :
                    if normalize_range is not None :
                        prob[normalize_range[0]:normalize_range[1]] += cut_prob[normalize_range[0]:normalize_range[1]] / np.sum(cut_prob[normalize_range[0]:normalize_range[1]])
                    else :
                        prob += cut_prob[:164] / np.sum(cut_prob[:164])
                else :
                    prob += cut_prob[:164]

            if agg_mode == 'avg' :
                prob /= float(len(seqs))
            elif agg_mode == 'pool' :
                prob /= np.sum(prob)

            for cut_prob in list(df_seq['cut_prob_pred'].values) :
                if normalize_probs :
                    if normalize_range is not None :
                        pred_prob[normalize_range[0]:normalize_range[1]] += cut_prob[normalize_range[0]:normalize_range[1]] / np.sum(cut_prob[normalize_range[0]:normalize_range[1]])
                    else :
                        pred_prob += cut_prob[:164] / np.sum(cut_prob[:164])
                else :
                    pred_prob += cut_prob[:164]

            if agg_mode == 'avg' :
                pred_prob /= float(len(seqs))
            elif agg_mode == 'pool' :
                pred_prob /= np.sum(pred_prob)

        wt_seq = wt_seqs[wt_max_index]
        n_samples = len(df_seq)
        wt_mean_logodds = wt_mean_logodds[wt_max_index]
        wt_avgpos = wt_avgpos[wt_max_index]
        wt_logloss = wt_logloss[wt_max_index]
        if override_mean_stats :
            n_samples = len(df_seqs)
            wt_mean_logodds = np.mean(wt_mean_logodds)
            wt_avgpos = np.mean(df_seqs['avgpos_true'])
            wt_logloss = np.mean(df_seqs['loss_logloss'])

        subexperiment = list(df_seqs.query("wt_seq == '" + wt_seq + "'")['subexperiment'].values)[0]
        opt_pwm = np.vstack([np.ones((1, 4)) * 0.25, find_wt_pwm(wt_seq, max_cut_pwm_dict[gene + '_' + subexperiment])])[:164, :]
        opt_prob = np.zeros(opt_pwm.shape[0])
        subexp_prob = find_wt_yhat(wt_seq, max_cut_pwm_dict[gene + '_' + subexperiment], max_cut_yhat_dict[gene + '_' + subexperiment])[:163]
        
        if normalize_probs :
            if normalize_range is not None :
                opt_prob[normalize_range[0]:normalize_range[1]] = subexp_prob[normalize_range[0]-1:normalize_range[1]-1] / np.sum(subexp_prob[normalize_range[0]-1:normalize_range[1]-1])
            else :
                opt_prob[1:] = subexp_prob[:163] / np.sum(subexp_prob[:163])
        else :
            opt_prob[1:] = subexp_prob[:163]
        
        if np.sum(opt_pwm) > 0 :
            for j in range(0, opt_pwm.shape[0]) :
                if np.sum(opt_pwm[j, :]) > 0 :
                    opt_pwm[j, :] /= np.sum(opt_pwm[j, :])

        #Slice according to seq trim index
        seqs = [seq[seq_trim_start: seq_trim_end] for seq in seqs]
        pwm = pwm[:, seq_trim_start: seq_trim_end]
        prob = prob[seq_trim_start: seq_trim_end]
        pred_prob = pred_prob[seq_trim_start: seq_trim_end]
        opt_pwm = opt_pwm[:, seq_trim_start: seq_trim_end]
        opt_prob = opt_prob[seq_trim_start: seq_trim_end]

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
            perc_ax[side_plot_i].plot(np.arange(len(human_cutprob)), human_cutprob, linewidth=2, color='black')
            perc_ax[side_plot_i].scatter([objective_pos], [human_cutprob[objective_pos]], s=70, c='red', alpha=1.0)
            
            if objective_pos <= 30 :
                perc_ax[side_plot_i].annotate('Objective', xy=(objective_pos, human_cutprob[objective_pos]), xycoords='data', xytext=(0.55, 0.8), fontsize=10, weight="bold", color='red', textcoords='axes fraction', arrowprops=dict(connectionstyle="arc3,rad=-.2", headlength=8, headwidth=8, shrink=0.05, width=1.5, color='black'))
            else :
                perc_ax[side_plot_i].annotate('Objective', xy=(objective_pos, human_cutprob[objective_pos]), xycoords='data', xytext=(0.55, 0.8), fontsize=10, weight="bold", color='red', textcoords='axes fraction', arrowprops=dict(connectionstyle="arc3,rad=.2", headlength=8, headwidth=8, shrink=0.05, width=1.5, color='black'))
            
            perc_ax[side_plot_i].axvline(x=0, linewidth=1.5, color='green', linestyle='--')
            perc_ax[side_plot_i].axvline(x=6, linewidth=1.5, color='green', linestyle='--')

        if plot_actual_pwm :
            
            l2, = cut_ax[2].plot(np.arange(plot_end - plot_start) + plot_start, prob[plot_start:plot_end], linewidth=3, linestyle='-', label='Observed', color='black', alpha=0.7)
            l1, = cut_ax[2].plot(np.arange(plot_end - plot_start) + plot_start, pred_prob[plot_start:plot_end], linewidth=3, linestyle='-', label='Predicted', color='red', alpha=0.7)
            
            if annotate_peaks :
                annot_text = str(int(round(prob[objective_pos + 50] * 100, 0))) + '% Cleavage'
                cut_ax[2].annotate(annot_text, xy=(objective_pos + 50, prob[objective_pos + 50]), xycoords='data', xytext=(-30, -5), ha='right', fontsize=10, weight="bold", color='black', textcoords='offset points', arrowprops=dict(connectionstyle="arc3,rad=-.1", headlength=8, headwidth=8, shrink=0.15, width=1.5, color='black'))
            
            plt.sca(cut_ax[2])

            plt.xlim((plot_start, plot_end))
            #plt.ylim((0, 2))
            plt.xticks([], [])
            plt.yticks([], [])
            plt.legend(handles=[l1, l2], fontsize=12, prop=dict(weight='bold'), frameon=False)
            plt.axis('off')
            
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

            if plot_mfe :
                mfe = list(df_seqs.query("wt_seq == '" + wt_seq + "'")['mfe'].values)[0]
                mfe_struct = ('X' * 56) + list(df_seqs.query("wt_seq == '" + wt_seq + "'")['struct'].values)[0] + ('X' * 100)
                
                for j in range(plot_start, plot_end) :
                    if mfe_struct[j] != 'X' :
                        letterAt(mfe_struct[j], j + 0.5, -fold_height, fold_height-0.05, logo_ax[2], color='black')
                
                annot_text = 'MFE = ' + str(round(mfe, 1))
                #logo_ax[2].annotate(annot_text, xy=(56, -fold_height/2), xycoords='data', xytext=(-30, 0), ha='right', fontsize=10, weight="bold", color='black', textcoords='offset points', arrowprops=dict(headlength=8, headwidth=8, shrink=0.15, width=1.5, color='black'))
                logo_ax[2].text(55, -fold_height/2 -0.05, annot_text, horizontalalignment='right', verticalalignment='center', color='black', fontsize=12, weight="bold")
                
            
            plt.sca(logo_ax[2])

            plt.xlim((plot_start, plot_end))
            if plot_mfe :
                plt.ylim((-fold_height-0.02, 2))
            else :
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
            
            l2, = cut_ax[3].plot(np.arange(plot_end - plot_start) + plot_start, prob[plot_start:plot_end], linewidth=3, linestyle='-', label='Observed', color='black', alpha=0.7)
            l1, = cut_ax[3].plot(np.arange(plot_end - plot_start) + plot_start, opt_prob[plot_start:plot_end], linewidth=3, linestyle='-', label='Predicted', color='red', alpha=0.7)
            
            if annotate_peaks :
                annot_text = str(int(round(prob[objective_pos + 50] * 100, 0))) + '% Cleavage'
                cut_ax[2].annotate(annot_text, xy=(objective_pos + 50, prob[objective_pos + 50]), xycoords='data', xytext=(-30, -5), ha='right', fontsize=10, weight="bold", color='black', textcoords='offset points', arrowprops=dict(connectionstyle="arc3,rad=-.1", headlength=8, headwidth=8, shrink=0.15, width=1.5, color='black'))
            
            plt.sca(cut_ax[3])

            plt.xlim((plot_start, plot_end))
            #plt.ylim((0, 2))
            plt.xticks([], [])
            plt.yticks([], [])
            plt.legend(handles=[l1, l2], fontsize=12, prop=dict(weight='bold'), frameon=False)
            plt.axis('off')
            
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

            if plot_mfe :
                mfe = list(df_seqs.query("wt_seq == '" + wt_seq + "'")['mfe'].values)[0]
                mfe_struct = ('X' * 56) + list(df_seqs.query("wt_seq == '" + wt_seq + "'")['struct'].values)[0] + ('X' * 100)
                
                for j in range(plot_start, plot_end) :
                    if mfe_struct[j] != 'X' :
                        letterAt(mfe_struct[j], j + 0.5, -fold_height, fold_height-0.05, logo_ax[2], color='black')
                
                annot_text = 'MFE = ' + str(round(mfe, 1))
                #logo_ax[2].annotate(annot_text, xy=(56, -fold_height/2), xycoords='data', xytext=(-30, 0), ha='right', fontsize=10, weight="bold", color='black', textcoords='offset points', arrowprops=dict(headlength=8, headwidth=8, shrink=0.15, width=1.5, color='black'))
                logo_ax[2].text(55, -fold_height/2 -0.05, annot_text, horizontalalignment='right', verticalalignment='center', color='black', fontsize=12, weight="bold")
                
            
            plt.sca(logo_ax[2])

            plt.xlim((plot_start, plot_end))
            if plot_mfe :
                plt.ylim((-fold_height-0.02, 2))
            else :
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

#Max Cut- optimized sequence PWMs (generated by SeqProp)
def load_max_cut_pwms() :
    file_path = 'max_cut_logos/'

    max_cut_pwm_dict = {}
    max_cut_yhat_dict = {}

    cut_list = ['60', '65', '70', '75', '80', '85', '90', '95', '100']
    cut_list_last = ['85', '90', '95', '100']


    #A
    for cut_pos_i in range(0, len(cut_list)) :
        cut_pos_str = cut_list[cut_pos_i]
        cut_pos = int(cut_pos_str)

        y_hats_1 = np.load(file_path + 'A_iter_4000/simple_' + cut_pos_str + '_max_class_max_score_1_images_cuthat.npy')
        opt_index_1 = np.argsort(np.sum(y_hats_1[:, cut_pos-1:cut_pos+2], axis=1), axis=0)[::-1]
        pwms_1 = np.load(file_path + 'A_iter_4000/simple_' + cut_pos_str + '_max_class_max_score_1_images_pwm.npy')[opt_index_1, :, :][:3, :, :]
        y_hats_1 = y_hats_1[opt_index_1, :][:3, :]

        y_hats_2 = np.load(file_path + 'simple_' + cut_pos_str + '_max_class_max_score_1_images_cuthat.npy')
        opt_index_2 = np.argsort(np.sum(y_hats_2[:, cut_pos-1:cut_pos+2], axis=1), axis=0)[::-1]
        pwms_2 = np.load(file_path + 'simple_' + cut_pos_str + '_max_class_max_score_1_images_pwm.npy')[opt_index_2, :, :][:2, :, :]
        y_hats_2 = y_hats_2[opt_index_2, :][:2, :]

        y_hats_ent = np.load(file_path + 'simple_' + cut_pos_str + '_ent_max_class_max_score_ent_1_images_cuthat.npy')
        opt_index_ent = np.argsort(np.sum(y_hats_ent[:, cut_pos-1:cut_pos+2], axis=1), axis=0)[::-1]
        pwms_ent = np.load(file_path + 'simple_' + cut_pos_str + '_ent_max_class_max_score_ent_1_images_pwm.npy')[opt_index_ent, :, :][:2, :, :]
        y_hats_ent = y_hats_ent[opt_index_ent, :][:2, :]

        max_cut_pwm_dict['simple_' + 'A' + '_' + str(cut_pos_str)] = np.concatenate([pwms_1, pwms_2], axis=0)
        max_cut_yhat_dict['simple_' + 'A' + '_' + str(cut_pos_str)] = np.concatenate([y_hats_1, y_hats_2], axis=0)
        
        max_cut_pwm_dict['simple_' + 'A_ent' + '_' + str(cut_pos_str)] = pwms_ent
        max_cut_yhat_dict['simple_' + 'A_ent' + '_' + str(cut_pos_str)] = y_hats_ent


    #A GGCC
    for cut_pos_i in range(0, len(cut_list_last)) :
        cut_pos_str = cut_list_last[cut_pos_i]

        y_hats_1 = np.load(file_path + 'A_GGCC_iter_4000/simple_' + cut_pos_str + '_GGCC_max_class_max_score_GGCC_1_images_cuthat.npy')
        opt_index_1 = np.argsort(np.sum(y_hats_1[:, cut_pos-1:cut_pos+2], axis=1), axis=0)[::-1]
        pwms_1 = np.load(file_path + 'A_GGCC_iter_4000/simple_' + cut_pos_str + '_GGCC_max_class_max_score_GGCC_1_images_pwm.npy')[opt_index_1, :, :][:3, :, :]

        y_hats_2 = np.load(file_path + 'simple_' + cut_pos_str + '_GGCC_max_class_max_score_GGCC_1_images_cuthat.npy')
        opt_index_2 = np.argsort(np.sum(y_hats_2[:, cut_pos-1:cut_pos+2], axis=1), axis=0)[::-1]
        pwms_2 = np.load(file_path + 'simple_' + cut_pos_str + '_GGCC_max_class_max_score_GGCC_1_images_pwm.npy')[opt_index_2, :, :][:2, :, :]

        y_hats_ent = np.load(file_path + 'simple_' + cut_pos_str + '_GGCC_ent_max_class_max_score_GGCC_ent_1_images_cuthat.npy')
        opt_index_ent = np.argsort(np.sum(y_hats_ent[:, cut_pos-1:cut_pos+2], axis=1), axis=0)[::-1]
        pwms_ent = np.load(file_path + 'simple_' + cut_pos_str + '_GGCC_ent_max_class_max_score_GGCC_ent_1_images_pwm.npy')[opt_index_ent, :, :][:2, :, :]

        max_cut_pwm_dict['simple_' + 'A_GGCC' + '_' + str(cut_pos_str)] = np.concatenate([pwms_1, pwms_2], axis=0)
        max_cut_yhat_dict['simple_' + 'A_GGCC' + '_' + str(cut_pos_str)] = np.concatenate([y_hats_1, y_hats_2], axis=0)
        
        max_cut_pwm_dict['simple_' + 'A_GGCC_ent' + '_' + str(cut_pos_str)] = pwms_ent
        max_cut_yhat_dict['simple_' + 'A_GGCC_ent' + '_' + str(cut_pos_str)] = y_hats_ent


    #AT
    for cut_pos_i in range(0, len(cut_list)) :
        cut_pos_str = cut_list[cut_pos_i]

        y_hats_1 = np.load(file_path + 'AT_iter_5000/simple_' + cut_pos_str + '_AT_max_class_max_score_1_images_cuthat.npy')
        opt_index_1 = np.argsort(np.sum(y_hats_1[:, cut_pos-1:cut_pos+2], axis=1), axis=0)[::-1]
        pwms_1 = np.load(file_path + 'AT_iter_5000/simple_' + cut_pos_str + '_AT_max_class_max_score_1_images_pwm.npy')[opt_index_1, :, :][:3, :, :]

        y_hats_2 = np.load(file_path + 'simple_' + cut_pos_str + '_AT_max_class_max_score_1_images_cuthat.npy')
        opt_index_2 = np.argsort(np.sum(y_hats_2[:, cut_pos-1:cut_pos+2], axis=1), axis=0)[::-1]
        pwms_2 = np.load(file_path + 'simple_' + cut_pos_str + '_AT_max_class_max_score_1_images_pwm.npy')[opt_index_2, :, :][:2, :, :]

        y_hats_ent = np.load(file_path + 'simple_' + cut_pos_str + '_AT_ent_max_class_max_score_ent_1_images_cuthat.npy')
        opt_index_ent = np.argsort(np.sum(y_hats_ent[:, cut_pos-1:cut_pos+2], axis=1), axis=0)[::-1]
        pwms_ent = np.load(file_path + 'simple_' + cut_pos_str + '_AT_ent_max_class_max_score_ent_1_images_pwm.npy')[opt_index_ent, :, :][:4, :, :]

        max_cut_pwm_dict['simple_' + 'AT' + '_' + str(cut_pos_str)] = np.concatenate([pwms_1, pwms_2], axis=0)
        max_cut_yhat_dict['simple_' + 'AT' + '_' + str(cut_pos_str)] = np.concatenate([y_hats_1, y_hats_2], axis=0)
        
        max_cut_pwm_dict['simple_' + 'AT_ent' + '_' + str(cut_pos_str)] = pwms_ent
        max_cut_yhat_dict['simple_' + 'AT_ent' + '_' + str(cut_pos_str)] = y_hats_ent



    #AT GGCC
    for cut_pos_i in range(0, len(cut_list_last)) :
        cut_pos_str = cut_list_last[cut_pos_i]

        y_hats_1 = np.load(file_path + 'AT_GGCC_strong_iter_5000/simple_' + cut_pos_str + '_AT_GGCC_max_class_max_score_GGCC_1_images_cuthat.npy')
        opt_index_1 = np.argsort(np.sum(y_hats_1[:, cut_pos-1:cut_pos+2], axis=1), axis=0)[::-1]
        pwms_1 = np.load(file_path + 'AT_GGCC_strong_iter_5000/simple_' + cut_pos_str + '_AT_GGCC_max_class_max_score_GGCC_1_images_pwm.npy')[opt_index_1, :, :][:3, :, :]

        y_hats_2 = np.load(file_path + 'simple_' + cut_pos_str + '_AT_GGCC_max_class_max_score_GGCC_1_images_cuthat.npy')
        opt_index_2 = np.argsort(np.sum(y_hats_2[:, cut_pos-1:cut_pos+2], axis=1), axis=0)[::-1]
        pwms_2 = np.load(file_path + 'simple_' + cut_pos_str + '_AT_GGCC_max_class_max_score_GGCC_1_images_pwm.npy')[opt_index_2, :, :][:2, :, :]

        y_hats_ent = np.load(file_path + 'simple_' + cut_pos_str + '_AT_GGCC_ent_max_class_max_score_GGCC_ent_1_images_cuthat.npy')
        opt_index_ent = np.argsort(np.sum(y_hats_ent[:, cut_pos-1:cut_pos+2], axis=1), axis=0)[::-1]
        pwms_ent = np.load(file_path + 'simple_' + cut_pos_str + '_AT_GGCC_ent_max_class_max_score_GGCC_ent_1_images_pwm.npy')[opt_index_ent, :, :][:4, :, :]

        max_cut_pwm_dict['simple_' + 'AT_GGCC' + '_' + str(cut_pos_str)] = np.concatenate([pwms_1, pwms_2], axis=0)
        max_cut_yhat_dict['simple_' + 'AT_GGCC' + '_' + str(cut_pos_str)] = np.concatenate([y_hats_1, y_hats_2], axis=0)
        
        max_cut_pwm_dict['simple_' + 'AT_GGCC_ent' + '_' + str(cut_pos_str)] = pwms_ent
        max_cut_yhat_dict['simple_' + 'AT_GGCC_ent' + '_' + str(cut_pos_str)] = y_hats_ent
        


    #A punish aruns
    for cut_pos_i in range(0, len(cut_list)) :
        cut_pos_str = cut_list[cut_pos_i]

        y_hats_1 = np.load(file_path + 'simple_' + cut_pos_str + '_punish_aruns_max_class_max_score_punish_aruns_1_images_cuthat.npy')
        opt_index_1 = np.argsort(np.sum(y_hats_1[:, cut_pos-1:cut_pos+2], axis=1), axis=0)[::-1]
        pwms_1 = np.load(file_path + 'simple_' + cut_pos_str + '_punish_aruns_max_class_max_score_punish_aruns_1_images_pwm.npy')[opt_index_1, :, :][:5, :, :]

        y_hats_ent = np.load(file_path + 'simple_' + cut_pos_str + '_punish_aruns_ent_max_class_max_score_punish_aruns_ent_1_images_cuthat.npy')
        opt_index_ent = np.argsort(np.sum(y_hats_ent[:, cut_pos-1:cut_pos+2], axis=1), axis=0)[::-1]
        pwms_ent = np.load(file_path + 'simple_' + cut_pos_str + '_punish_aruns_ent_max_class_max_score_punish_aruns_ent_1_images_pwm.npy')[opt_index_ent, :, :][:4, :, :]

        max_cut_pwm_dict['simple_' + 'A_aruns' + '_' + str(cut_pos_str)] = pwms_1
        max_cut_yhat_dict['simple_' + 'A_aruns' + '_' + str(cut_pos_str)] = y_hats_1
        
        max_cut_pwm_dict['simple_' + 'A_aruns_ent' + '_' + str(cut_pos_str)] = pwms_ent
        max_cut_yhat_dict['simple_' + 'A_aruns_ent' + '_' + str(cut_pos_str)] = y_hats_ent


    #A GGCC punish aruns
    for cut_pos_i in range(0, len(cut_list_last)) :
        cut_pos_str = cut_list_last[cut_pos_i]

        y_hats_1 = np.load(file_path + 'simple_' + cut_pos_str + '_GGCC_punish_aruns_max_class_max_score_GGCC_punish_aruns_1_images_cuthat.npy')
        opt_index_1 = np.argsort(np.sum(y_hats_1[:, cut_pos-1:cut_pos+2], axis=1), axis=0)[::-1]
        pwms_1 = np.load(file_path + 'simple_' + cut_pos_str + '_GGCC_punish_aruns_max_class_max_score_GGCC_punish_aruns_1_images_pwm.npy')[opt_index_1, :, :][:5, :, :]

        y_hats_ent = np.load(file_path + 'simple_' + cut_pos_str + '_GGCC_punish_aruns_ent_max_class_max_score_GGCC_punish_aruns_ent_1_images_cuthat.npy')
        opt_index_ent = np.argsort(np.sum(y_hats_ent[:, cut_pos-1:cut_pos+2], axis=1), axis=0)[::-1]
        pwms_ent = np.load(file_path + 'simple_' + cut_pos_str + '_GGCC_punish_aruns_ent_max_class_max_score_GGCC_punish_aruns_ent_1_images_pwm.npy')[opt_index_ent, :, :][:4, :, :]

        max_cut_pwm_dict['simple_' + 'A_GGCC_aruns' + '_' + str(cut_pos_str)] = pwms_1
        max_cut_yhat_dict['simple_' + 'A_GGCC_aruns' + '_' + str(cut_pos_str)] = y_hats_1
        
        max_cut_pwm_dict['simple_' + 'A_GGCC_aruns_ent' + '_' + str(cut_pos_str)] = pwms_ent
        max_cut_yhat_dict['simple_' + 'A_GGCC_aruns_ent' + '_' + str(cut_pos_str)] = y_hats_ent

    return max_cut_pwm_dict, max_cut_yhat_dict

#Max Cut SNV Helper Functions

def mut_map_fold_snvs(df_gene, gene_name, experiment, mode, figsize=(12, 3), delta_column='delta_logodds', mark_pathogenic=False, mark_benign=False, mark_undetermined=False, border_eta=0.085, seq_trim_start=0, seq_trim_end=164, plot_start=0, plot_end=164, cut_downscaling=0.5, pas_downscale_mode='frac', fig_name=None, fig_dpi=300) :

    mut_map = np.zeros((4, 164))
    ref_seq = df_gene['wt_seq'].values[0]

    for _, row in df_gene.iterrows() :
        snv_pos = row['snv_pos']
        
        if row['wt_seq'] != ref_seq :
            continue

        delta_logodds_true = row[delta_column]
        if np.isnan(delta_logodds_true) :
            delta_logodds_true = 0

        base = 0
        if row['master_seq'][snv_pos] == 'A' :
            base = 0
        elif row['master_seq'][snv_pos] == 'C' :
            base = 1
        elif row['master_seq'][snv_pos] == 'G' :
            base = 2
        elif row['master_seq'][snv_pos] == 'T' :
            base = 3

        mut_map[3-base, snv_pos] = delta_logodds_true
    
    obj_pos = int(experiment[0].split("_")[-1]) + 1
    mut_map[:, obj_pos] = mut_map[:, obj_pos] * cut_downscaling
    
    #Slice according to seq trim index
    ref_seq = ref_seq[seq_trim_start: seq_trim_end]
    mut_map = mut_map[:, seq_trim_start: seq_trim_end]

    fig = plt.figure(figsize=figsize) 
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax = [ax0, ax1]

    bias = np.max(np.sum(mut_map[:, :], axis=0)) / 3.0 + 0.5
    max_score = np.min(np.sum(mut_map[:, :], axis=0)) / 3.0 * -1 + bias

    for i in range(plot_start, plot_end) :
        mutability_score = np.sum(mut_map[:, i]) / 3.0 * -1 + bias
        letterAt(ref_seq[i], i + 0.5, 0, mutability_score, ax[0])

    ax[0].plot([0, mut_map.shape[1]], [bias, bias], color='black', linestyle='--')
    
    plt.sca(ax[0])
    plt.yticks([0.5, bias, max_score], [round(bias - 0.5, 2), 0, round((max_score - bias) * -1, 2)], fontsize=16)
    plt.xticks(fontsize=16)
    plt.xlim((plot_start, plot_end)) 
    plt.ylim((0, max_score)) 
    plt.tight_layout()

    
    pcm = ax[1].pcolor(mut_map, cmap='RdBu_r', vmin=-np.abs(mut_map).max(), vmax=np.abs(mut_map).max())
    #fig.colorbar(pcm, ax=ax[1])
    
    plt.sca(ax[1])

    ref_seq_list = []
    for c in ref_seq :
        ref_seq_list.append(c)
    plt.xticks(np.arange(len(ref_seq)) + 0.5, ref_seq_list)
    plt.xticks([], [])

    plt.yticks([0.5, 1.5, 2.5, 3.5], ['T', 'G', 'C', 'A'], fontsize=16)
    plt.axis([plot_start, plot_end, 0, 4])
    
    plt.gca().xaxis.tick_top()

    #plt.savefig(name + '.svg', bbox_inches='tight')
    #plt.savefig(name + '.png', bbox_inches='tight')
    plt.tight_layout()
    
    if fig_name is not None :
        plt.savefig(fig_name + '.png', transparent=True, dpi=fig_dpi)
        plt.savefig(fig_name + '.svg')
        plt.savefig(fig_name + '.eps')
    plt.show()

def struct_map_fold_snvs(df_gene, gene_name, experiment, mode, figsize=(12, 3), delta_column='delta_logodds', mark_pathogenic=False, mark_benign=False, mark_undetermined=False, border_eta=0.085, seq_trim_start=0, seq_trim_end=164, plot_start=0, plot_end=164, cut_downscaling=0.5, pas_downscale_mode='frac', fig_name=None, fig_dpi=300) :

    mut_map = np.zeros((3, 164))
    count_map = np.zeros((3, 164))
    ref_seq = df_gene['wt_seq'].values[0]

    for _, row in df_gene.iterrows() :
        snv_pos = row['snv_pos']
        
        if row['wt_seq'] != ref_seq :
            continue

        delta_logodds_true = row[delta_column]
        if np.isnan(delta_logodds_true) :
            delta_logodds_true = 0
        
        base = 0
        if row['master_seq'][snv_pos] == '(' :
            base = 0
        elif row['master_seq'][snv_pos] == '.' :
            base = 1
        elif row['master_seq'][snv_pos] == ')' :
            base = 2

        mut_map[2-base, snv_pos] += delta_logodds_true
        count_map[2-base, snv_pos] += delta_logodds_true
    
    mut_map[count_map > 0] /= count_map[count_map > 0]
    
    obj_pos = int(experiment[0].split("_")[-1]) + 1
    mut_map[:, obj_pos] = mut_map[:, obj_pos] * cut_downscaling
    
    #Slice according to seq trim index
    ref_seq = ref_seq[seq_trim_start: seq_trim_end]
    mut_map = mut_map[:, seq_trim_start: seq_trim_end]

    fig = plt.figure(figsize=figsize) 
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax = [ax0, ax1]

    bias = np.max(np.sum(mut_map[:, :], axis=0)) / 2.0 + 0.5
    max_score = np.min(np.sum(mut_map[:, :], axis=0)) / 2.0 * -1 + bias

    for i in range(plot_start, plot_end) :
        mutability_score = np.sum(mut_map[:, i]) / 2.0 * -1 + bias
        letterAt(ref_seq[i], i + 0.5, 0, mutability_score, ax[0])

    ax[0].plot([0, mut_map.shape[1]], [bias, bias], color='black', linestyle='--')
    
    plt.sca(ax[0])
    plt.yticks([0.5, bias, max_score], [round(bias - 0.5, 2), 0, round((max_score - bias) * -1, 2)], fontsize=16)
    plt.xticks(fontsize=16)
    plt.xlim((plot_start, plot_end)) 
    plt.ylim((0, max_score)) 
    plt.tight_layout()

    
    pcm = ax[1].pcolor(mut_map, cmap='RdBu_r', vmin=-np.abs(mut_map).max(), vmax=np.abs(mut_map).max())
    #fig.colorbar(pcm, ax=ax[1])
    
    plt.sca(ax[1])

    ref_seq_list = []
    for c in ref_seq :
        ref_seq_list.append(c)
    plt.xticks(np.arange(len(ref_seq)) + 0.5, ref_seq_list)
    plt.xticks([], [])

    plt.yticks([0.5, 1.5, 2.5], [')', '.', '('], fontsize=16)
    plt.axis([plot_start, plot_end, 0, 3])
    
    plt.gca().xaxis.tick_top()

    #plt.savefig(name + '.svg', bbox_inches='tight')
    #plt.savefig(name + '.png', bbox_inches='tight')
    plt.tight_layout()
    
    if fig_name is not None :
        plt.savefig(fig_name + '.png', transparent=True, dpi=fig_dpi)
        plt.savefig(fig_name + '.svg')
        plt.savefig(fig_name + '.eps')
    plt.show()

