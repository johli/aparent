import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io as spio

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter

from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import pearsonr

from scipy.stats import norm

import operator

#Two-sided proportion difference test
def differential_prop_test(count_1, total_count_1, count_2, total_count_2) :
    p1_hat = count_1 / total_count_1
    p2_hat = count_2 / total_count_2
    p_hat = (count_1 + count_2) / (total_count_1 + total_count_2)
    
    z = (p1_hat - p2_hat) / np.sqrt(p_hat * (1. - p_hat) * (1. / total_count_1 + 1. / total_count_2))
    z_abs = np.abs(z)
    
    z_rv = norm()
    p_val = 2. * z_rv.sf(z_abs)
    log_p_val = np.log(2) + z_rv.logsf(z_abs)
    
    return p_val, log_p_val

#Join PAS dataframe with predicted PAS dataframe
def join_pas_dataframes(native_dict, pred_dict, leslie_tissue_index, apadb_tissue_index) :
	df = native_dict['df']
	leslie_cleavage_count_dict = { cell_type : native_dict[cell_type][:, 105 + 20 : 105 + 205] for cell_type in native_dict if cell_type != 'df' }

	leslie_cleavage_prob_dict = {
	    cell_type : leslie_cleavage_count_dict[cell_type] / np.clip(leslie_cleavage_count_dict[cell_type].sum(axis=-1).reshape(-1, 1), a_min=0.1, a_max=None)
	    for cell_type in leslie_cleavage_count_dict
	}

	pred_df = pred_dict['native_df']
	pred_cleavage_prob = pred_dict['cut_prob'][:, 20:-1]

	#Join predictions onto master dataframe
	df['row_index'] = np.arange(len(df), dtype=np.int)
	pred_df['row_index'] = np.arange(len(pred_df), dtype=np.int)

	df = df.join(pred_df.set_index('gene_id'), on='gene_id', how='inner', rsuffix='_pred').copy().reset_index(drop=True)

	for cell_type in leslie_cleavage_prob_dict :
	    leslie_cleavage_count_dict[cell_type] = leslie_cleavage_count_dict[cell_type][np.ravel(df['row_index'].values), :]
	    leslie_cleavage_prob_dict[cell_type] = leslie_cleavage_prob_dict[cell_type][np.ravel(df['row_index'].values), :]

	pred_cleavage_prob = pred_cleavage_prob[np.ravel(df['row_index_pred'].values), :]

	#Build isoform count matrices (Rows = PAS sequences / Cols = Cell types)

	leslie_isoform_count = np.zeros((len(df), len(leslie_tissue_index)))
	for i, cell_type in enumerate(leslie_tissue_index.tolist()) :
	    leslie_isoform_count[:, i] = df['leslie_count_apadb_region_' + cell_type]

	apadb_isoform_count = np.zeros((len(df), len(apadb_tissue_index)))
	for i, cell_type in enumerate(apadb_tissue_index.tolist()) :
	    apadb_isoform_count[:, i] = df['apadb_count_' + cell_type]


	return df, leslie_isoform_count, apadb_isoform_count, leslie_cleavage_count_dict, leslie_cleavage_prob_dict, pred_cleavage_prob

#Join APA dataframe with predicted PAS dataframe
def join_apa_dataframes(pair_dict, pair_pred_dict, leslie_tissue_index, apadb_tissue_index) :
	pair_df = pair_dict['df_pair']
	leslie_cleavage_count_prox_dict = { cell_type : pair_dict[cell_type][:, 105 + 20 : 105 + 205] for cell_type in pair_dict if '_prox' in cell_type }
	leslie_cleavage_count_dist_dict = { cell_type : pair_dict[cell_type][:, 105 + 20 : 105 + 205] for cell_type in pair_dict if '_dist' in cell_type }

	leslie_cleavage_prob_prox_dict = {
	    cell_type : leslie_cleavage_count_prox_dict[cell_type] / np.clip(leslie_cleavage_count_prox_dict[cell_type].sum(axis=-1).reshape(-1, 1), a_min=0.1, a_max=None)
	    for cell_type in leslie_cleavage_count_prox_dict
	}
	leslie_cleavage_prob_dist_dict = {
	    cell_type : leslie_cleavage_count_dist_dict[cell_type] / np.clip(leslie_cleavage_count_dist_dict[cell_type].sum(axis=-1).reshape(-1, 1), a_min=0.1, a_max=None)
	    for cell_type in leslie_cleavage_count_dist_dict
	}

	pair_pred_df = pair_pred_dict['native_df']
	pred_cleavage_prob_prox = pair_pred_dict['cut_prox'][:, 20:]
	pred_cleavage_prob_dist = pair_pred_dict['cut_dist'][:, 20:]

	#Join predictions onto master dataframe
	pair_df['row_index'] = np.arange(len(pair_df), dtype=np.int)
	pair_pred_df['row_index'] = np.arange(len(pair_pred_df), dtype=np.int)

	pair_df = pair_df.join(pair_pred_df.set_index('gene_id'), on='gene_id', how='inner', rsuffix='_pred').copy().reset_index(drop=True)

	for cell_type in leslie_cleavage_prob_prox_dict :
	    leslie_cleavage_count_prox_dict[cell_type] = leslie_cleavage_count_prox_dict[cell_type][np.ravel(pair_df['row_index'].values), :]
	    leslie_cleavage_prob_prox_dict[cell_type] = leslie_cleavage_prob_prox_dict[cell_type][np.ravel(pair_df['row_index'].values), :]
	for cell_type in leslie_cleavage_prob_dist_dict :
	    leslie_cleavage_count_dist_dict[cell_type] = leslie_cleavage_count_dist_dict[cell_type][np.ravel(pair_df['row_index'].values), :]
	    leslie_cleavage_prob_dist_dict[cell_type] = leslie_cleavage_prob_dist_dict[cell_type][np.ravel(pair_df['row_index'].values), :]

	pred_cleavage_prob_prox = pred_cleavage_prob_prox[np.ravel(pair_df['row_index_pred'].values), :]
	pred_cleavage_prob_dist = pred_cleavage_prob_dist[np.ravel(pair_df['row_index_pred'].values), :]

	return pair_df, leslie_cleavage_count_prox_dict, leslie_cleavage_prob_prox_dict, leslie_cleavage_count_dist_dict, leslie_cleavage_prob_dist_dict, pred_cleavage_prob_prox, pred_cleavage_prob_dist

#Basic statistics plotting functions
def plot_cut_2mers(df, cell_type, cleavage_mat, seq_column='seq') :
    cut_mer2 = {}

    cx = sp.coo_matrix(cleavage_mat)

    for i,j,v in zip(cx.row, cx.col, cx.data) :
        seq = df.iloc[i][seq_column]

        mer2 = seq[j-1:j+1]
        if mer2 not in cut_mer2 :
            cut_mer2[mer2] = 0
        cut_mer2[mer2] += 1

    cut_mer2_sorted = sorted(cut_mer2.items(), key=operator.itemgetter(1))

    mer2_list = []
    mer2_vals = []
    for i in range(0, len(cut_mer2_sorted)) :
        mer2_list.append(cut_mer2_sorted[i][0])
        mer2_vals.append(cut_mer2_sorted[i][1])

    f = plt.figure(figsize=(6, 4))

    plt.bar(mer2_list, mer2_vals, color='black')

    plt.title('Proximal cleavage dinuc. (Leslie ' + cell_type + ')', fontsize=14)
    plt.xlabel('Dinucleotide', fontsize=14)
    plt.ylabel('Read count', fontsize=14)

    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14)

    plt.tight_layout()
    plt.show()

def plot_read_coverage_distribution(isoform_count, source, read_count_marks=[100, 1000]) :
	avg_isoform_count = np.mean(isoform_count, axis=1)
	sum_isoform_count = np.sum(isoform_count, axis=1)
	n_events = avg_isoform_count.shape[0]

	sort_index = np.argsort(avg_isoform_count)[::-1]

	f = plt.figure(figsize=(10, 4))

	ls = []

	l_avg, = plt.plot(np.arange(n_events), avg_isoform_count[sort_index], c='black', linewidth=2, label='Avg')
	ls.append(l_avg)

	sum_sort_index = np.argsort(sum_isoform_count)[::-1]
	l_sum, = plt.plot(np.arange(n_events), sum_isoform_count[sort_index], c='orange', linewidth=2, label='Pooled')
	ls.append(l_sum)

	#Avg marks
	l_mark_coords = []
	for read_count_mark in read_count_marks :
	    n_marked_events = np.sum(avg_isoform_count >= read_count_mark)
	    l_mark_coords.append(n_marked_events)
	    l_mark, = plt.plot([n_marked_events, n_marked_events], [0, avg_isoform_count[sort_index][n_marked_events]], color='black', linestyle='--', linewidth=2, label='RC >= ' + str(read_count_mark))
	    
	    ls.append(l_mark)
	    
	    
	    n_marked_events = np.sum(sum_isoform_count >= read_count_mark)
	    l_mark_coords.append(n_marked_events)
	    l_mark, = plt.plot([n_marked_events, n_marked_events], [0, sum_isoform_count[sum_sort_index][n_marked_events]], color='orange', linestyle='--', linewidth=2, label='RC >= ' + str(read_count_mark))
	    
	    ls.append(l_mark)

	plt.xticks(l_mark_coords, l_mark_coords, fontsize=16, rotation=45)
	plt.xlabel('PolyA sites', fontsize=16)

	plt.yticks(fontsize=16)
	plt.ylabel('Read count', fontsize=16)

	plt.ylim(0, 1000)

	plt.legend(handles=ls, fontsize=14)
	plt.title("Read depth across pA sites (" + source + ")", fontsize=14)

	plt.tight_layout()
	plt.show()

def plot_tissue_read_count_histo(df, source, tissue_index, n_rows=4, n_cols=5) :

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    width = 0.35

    site_types = ['UTR5', 'Intron', 'Exon', 'UTR3', 'Extension']

    for row_i in range(0, n_rows) :
        for col_i in range(0, n_cols) :
            ax[row_i, col_i].axis('off')

    for tissue_i in range(0, len(tissue_index)) :
        tissue = tissue_index[tissue_i]

        row_i = int(tissue_i / n_cols)
        col_i = int(tissue_i % n_cols)

        ax[row_i, col_i].axis('on')

        site_type_i = 0

        read_counts = np.zeros(len(site_types))

        for site_type in site_types :
            read_counts[site_type_i] = np.sum(df.query("site_type == '" + site_type + "'")[source + '_count_' + tissue])
            site_type_i += 1

        p_color = 'r'
        if tissue == 'pooled' :
            p_color = 'darkblue'

        p1 = ax[row_i, col_i].bar(np.arange(len(site_types)), read_counts / (10.**6), width, color=p_color)

        ax[row_i, col_i].set_title(tissue, fontsize=14)

        ax[row_i, col_i].set_xticks(np.arange(len(site_types)) + width / 2)
        ax[row_i, col_i].set_xticklabels(site_types)
        ax[row_i, col_i].tick_params(axis='x', which='major', labelsize=12, rotation=45)
        ax[row_i, col_i].tick_params(axis='x', which='minor', labelsize=12, rotation=45)

        max_read_count = np.max(read_counts / (10.**6))
        ax[row_i, col_i].set_yticks([0.0, max_read_count * 0.5, max_read_count * 1.0])
        ax[row_i, col_i].yaxis.set_major_formatter(FormatStrFormatter('%.1fM'))
        ax[row_i, col_i].tick_params(axis='y', which='major', labelsize=12)
        ax[row_i, col_i].tick_params(axis='y', which='minor', labelsize=12)


    plt.tight_layout()
    plt.show()

def plot_site_type_fractions(df, source, tissue_index) :

    fig = plt.figure(figsize=(8, 6))

    site_types = ['Intron']#['Intron', 'Exon', 'UTR3']

    type_frac = np.zeros((len(site_types), len(tissue_index)))

    for tissue_i in range(0, len(tissue_index)) :
        tissue = tissue_index[tissue_i]

        site_type_i = 0
        for site_type in site_types :

            type_count = np.ravel(df.query("site_type == '" + site_type + "'")[source + '_count_' + tissue])
            all_type_count = np.ravel(df[source + '_count_' + tissue])

            type_frac[site_type_i, tissue_i] = np.sum(type_count) / np.sum(all_type_count)

            site_type_i += 1


    sort_index = np.argsort(np.ravel(type_frac[0, :]))[::-1]

    ls = []
    site_color = ['darkgreen', 'darkred', 'darkblue']

    site_type_i = 0
    for site_type in site_types :
        l1, = plt.plot(np.arange(len(tissue_index)), type_frac[site_type_i, sort_index], c=site_color[site_type_i], label=site_type, linewidth=3)

        ls.append(l1)

        site_type_i += 1

    plt.xticks(np.arange(len(tissue_index)), tissue_index[sort_index], fontsize=16, rotation=45)

    plt.ylim(0, 0.051)
    plt.yticks([0.0, 0.025, 0.05], [0, 2.5, 5], fontsize=16)
    plt.ylabel('% Intron read counts', fontsize=16)

    plt.legend(handles=ls, fontsize=16)

    plt.tight_layout()
    plt.show()