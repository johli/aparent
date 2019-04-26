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


#Get differential site usage statistics across cell types
def get_differential_usage_statistics(df, source_data, tissue_index, suffix_index, site_nums, site_types, pseudo_count, min_total_count, only_differentials, use_logodds, fixed_isoform_difference=0.25, total_count_str='total_count') :
    r2_map = np.zeros((len(tissue_index), len(tissue_index)))
    r2_map[np.arange(len(tissue_index)), np.arange(len(tissue_index))] = 1
    mae_map = np.zeros((len(tissue_index), len(tissue_index)))
    fixed_isoform_diff_map = np.zeros((len(tissue_index), len(tissue_index)))

    for tissue_i in range(0, len(tissue_index)) :
        tissue_1 = tissue_index[tissue_i]
        source_data_1 = source_data
        special_mode_1 = suffix_index[tissue_i]

        for tissue_j in range(tissue_i + 1, len(tissue_index)) :
            tissue_2 = tissue_index[tissue_j]
            source_data_2 = source_data
            special_mode_2 = suffix_index[tissue_j]

            count_col_1 = source_data_1 + '_count' + special_mode_1 + '_' + tissue_1 + ('_prox' if total_count_str == 'pair_count' else '')
            count_col_2 = source_data_2 + '_count' + special_mode_2 + '_' + tissue_2 + ('_prox' if total_count_str == 'pair_count' else '')
            total_count_col_1 = source_data_1 + '_' + total_count_str + special_mode_1 + '_' + tissue_1
            total_count_col_2 = source_data_2 + '_' + total_count_str + special_mode_2 + '_' + tissue_2
            df_to_use = df.query(total_count_col_1 + " >= " + str(min_total_count) + " and " + total_count_col_2 + " >= " + str(min_total_count))
            df_to_use = df_to_use.loc[df_to_use['site_type' + ('_prox' if total_count_str == 'pair_count' else '')].isin(site_types)]
            if site_nums is not None :
                df_to_use = df_to_use.loc[df_to_use['sitenum' + ('_prox' if total_count_str == 'pair_count' else '')].isin(site_nums)]

            if only_differentials :
                df_to_use = df_to_use.query(count_col_1 + " != " + total_count_col_1 + " and " + count_col_2 + " != " + total_count_col_2)
                df_to_use = df_to_use.query(count_col_1 + " != 0 and " + count_col_2 + " != 0")

            true_metric_tissue_1 = (df_to_use[count_col_1] + pseudo_count) / (df_to_use[total_count_col_1] + 2. * pseudo_count)
            true_metric_tissue_2 = (df_to_use[count_col_2] + pseudo_count) / (df_to_use[total_count_col_2] + 2. * pseudo_count)

            if use_logodds :
                true_metric_tissue_1 = np.log(true_metric_tissue_1 / (1. - true_metric_tissue_1))
                true_metric_tissue_2 = np.log(true_metric_tissue_2 / (1. - true_metric_tissue_2))

            pearson_r, _ = pearsonr(true_metric_tissue_1, true_metric_tissue_2)
            pearson_r2 = round(pearson_r * pearson_r, 4)

            r2_map[tissue_i, tissue_j] = pearson_r2
            r2_map[tissue_j, tissue_i] = pearson_r2
            
            mae = np.mean(np.abs(true_metric_tissue_1 - true_metric_tissue_2))
            mae_map[tissue_i, tissue_j] = mae
            
            isoform_diff_abs = np.abs(true_metric_tissue_1 - true_metric_tissue_2)
            fixed_isoform_diff_map[tissue_i, tissue_j] = float(len(np.nonzero(isoform_diff_abs > fixed_isoform_difference)[0])) / float(len(isoform_diff_abs))

    return r2_map, mae_map, fixed_isoform_diff_map

#Differential site usage heatmap (across cell types)
def plot_differential_usage_heatmap(r2_map, source_data, tissue_index) :
    f = plt.figure(figsize=(8, 8))

    plt.imshow(r2_map, cmap='Reds')

    plt.xticks(np.arange(len(tissue_index)), tissue_index, fontsize=14, rotation=90)
    plt.yticks(np.arange(len(tissue_index)), tissue_index, fontsize=14, rotation=0)

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)

    plt.title('Min / Mean R^2 = ' + str(round(np.min(r2_map), 2)) + ' / ' + str(round(np.mean(r2_map), 2)) + ' (' + source_data + ')', fontsize=14)

    plt.tight_layout()
    plt.show()

#Mean absolute isoform proportion difference (across cell types)
def plot_mean_absolute_difference(mae_map, tissue_index, top_n=20) :
    tissue_pair_list = []
    mae_list = []
    for i in range(0, len(tissue_index)) :
        for j in range(i + 1, len(tissue_index)) :
            tissue_pair_list.append(tissue_index[i] + ' vs. ' + tissue_index[j])
            mae_list.append(mae_map[i, j])
    
    tissue_pairs = np.array(tissue_pair_list, dtype=np.object)
    maes = np.array(mae_list)

    sort_index = np.argsort(maes)[::-1]
    tissue_pairs = tissue_pairs[sort_index]
    maes = maes[sort_index]
    
    f = plt.figure(figsize=(12, 6))
    
    plt.plot(np.arange(top_n), maes[:top_n], color='black', linewidth=2)
    
    plt.xticks(np.arange(top_n), tissue_pairs, fontsize=14, rotation=90)
    plt.yticks(fontsize=14)
    
    plt.xlabel("Tissue/Cell type pairs", fontsize=14)
    plt.ylabel("Mean Difference in pPAS Usage", fontsize=14)
    
    plt.tight_layout()
    plt.show()

#Fraction of sites with high differential usage (across cell types)
def plot_fraction_of_isoform_difference(fixed_isoform_diff_map, tissue_index, fixed_isoform_difference, top_n=20) :
    tissue_pair_list = []
    fixed_diff_list = []
    for i in range(0, len(tissue_index)) :
        for j in range(i + 1, len(tissue_index)) :
            tissue_pair_list.append(tissue_index[i] + ' vs. ' + tissue_index[j])
            fixed_diff_list.append(fixed_isoform_diff_map[i, j])
    
    tissue_pairs = np.array(tissue_pair_list, dtype=np.object)
    fixed_diffs = np.array(fixed_diff_list)
    
    sort_index = np.argsort(fixed_diffs)[::-1]
    tissue_pairs = tissue_pairs[sort_index]
    fixed_diffs = fixed_diffs[sort_index]
    
    f = plt.figure(figsize=(12, 6))
    
    plt.plot(np.arange(top_n), fixed_diffs[:top_n], color='black', linewidth=2)
    
    plt.xticks(np.arange(top_n), tissue_pairs, fontsize=14, rotation=90)
    plt.yticks(fontsize=14)
    
    plt.xlabel("Tissue/Cell type pairs", fontsize=14)
    plt.ylabel("% PASs with Diff > " + str(fixed_isoform_difference), fontsize=14)
    
    plt.tight_layout()
    plt.show()

#Differential total site usage analysis, individual scatter
def plot_individual_differential_scatter(df, tissue_1_info, tissue_2_info, site_nums, site_types, pseudo_count, min_total_count, only_differentials, use_logodds, total_count_str='total_count', color_significant_sites=True, alpha_confidence=10**(-10)) : 
    color_by_sitenum = False

    [source_data_1, tissue_1, special_mode_1] = tissue_1_info
    [source_data_2, tissue_2, special_mode_2] = tissue_2_info
    
    count_col_1 = source_data_1 + '_count' + special_mode_1 + '_' + tissue_1 + ('_prox' if total_count_str == 'pair_count' else '')
    count_col_2 = source_data_2 + '_count' + special_mode_2 + '_' + tissue_2 + ('_prox' if total_count_str == 'pair_count' else '')
    total_count_col_1 = source_data_1 + '_' + total_count_str + special_mode_1 + '_' + tissue_1
    total_count_col_2 = source_data_2 + '_' + total_count_str + special_mode_2 + '_' + tissue_2
    df_to_use = df.query(total_count_col_1 + " >= " + str(min_total_count) + " and " + total_count_col_2 + " >= " + str(min_total_count))
    df_to_use = df_to_use.loc[df_to_use['site_type' + ('_prox' if total_count_str == 'pair_count' else '')].isin(site_types)]
    if site_nums is not None :
        df_to_use = df_to_use.loc[df_to_use['sitenum' + ('_prox' if total_count_str == 'pair_count' else '')].isin(site_nums)]

    if only_differentials :
        df_to_use = df_to_use.query(count_col_1 + " != " + total_count_col_1 + " and " + count_col_2 + " != " + total_count_col_2)
        df_to_use = df_to_use.query(count_col_1 + " != 0 and " + count_col_2 + " != 0")

    true_metric_tissue_1 = (df_to_use[count_col_1] + pseudo_count) / (df_to_use[total_count_col_1] + 2. * pseudo_count)
    true_metric_tissue_2 = (df_to_use[count_col_2] + pseudo_count) / (df_to_use[total_count_col_2] + 2. * pseudo_count)

    if use_logodds :
        true_metric_tissue_1 = np.log(true_metric_tissue_1 / (1. - true_metric_tissue_1))
        true_metric_tissue_2 = np.log(true_metric_tissue_2 / (1. - true_metric_tissue_2))

    r_val, _ = pearsonr(true_metric_tissue_1, true_metric_tissue_2)

    f = plt.figure(figsize=(5, 5))

    if not (color_by_sitenum or color_significant_sites) :
        plt.scatter(true_metric_tissue_1, true_metric_tissue_2, alpha=0.25, s=5, c='black')
    elif color_significant_sites :
        p, log_p = differential_prop_test(np.ravel(df_to_use[count_col_1]) + pseudo_count, np.ravel(df_to_use[total_count_col_1]) + 2. * pseudo_count, np.ravel(df_to_use[count_col_2]) + pseudo_count, np.ravel(df_to_use[total_count_col_2]) + 2. * pseudo_count)
        
        plt.scatter(true_metric_tissue_1[p >= alpha_confidence], true_metric_tissue_2[p >= alpha_confidence], alpha=0.25, s=5, c='black')
        plt.scatter(true_metric_tissue_1[p < alpha_confidence], true_metric_tissue_2[p < alpha_confidence], alpha=0.5, s=5, c='red')
    elif color_by_sitenum :
        sitenums = np.ravel(df_to_use['sitenum'].values)
        for sitenum in np.unique(sitenums) :
            if np.any(sitenums == sitenum) :
                plt.scatter(true_metric_tissue_1[sitenums == sitenum], true_metric_tissue_2[sitenums == sitenum], label='Site ' + str(sitenum), alpha=0.25, s=5)

        plt.legend(fontsize=14, frameon=True, framealpha=0.5, loc='lower right')

    plt.xlabel(tissue_1.replace('_', ' ') + ' pPAS Usage (' + source_data_1 + ')', fontsize=18)
    plt.ylabel(tissue_2.replace('_', ' ') + ' pPAS Usage (' + source_data_2 + ')', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if not use_logodds :
        plt.xlim(0, 1)
        plt.ylim(0, 1)

    plt.title('n = ' + str(len(df_to_use)) + ', R^2 = ' + str(round(r_val * r_val, 2)), fontsize=18)

    plt.tight_layout()
    plt.show()

#Mean of normalized cleavage counts, per cell type
def plot_cut_distributions(df, tissue_index, cleavage_count_matrix_dict, site_types, plot_tissue_specific=True, plot_pooled=False, plot_predictions=False) :
    min_pos = 25
    max_pos = 150
    pas_offset = 49#50

    min_read_count = 10

    for site_type_i, site_type in enumerate(site_types) :

        fig = plt.figure(figsize=(6, 4))
        ls = []

        n_events = np.zeros(len(tissue_index))
        n_reads = np.zeros(len(tissue_index))

        cum_cleavage_distribs = np.zeros((185, len(tissue_index)))
        
        for tissue_i in range(0, len(tissue_index)) :
            tissue = tissue_index[tissue_i]
            cleavage_counts = np.array(cleavage_count_matrix_dict[tissue].todense())
            
            filter_index = np.nonzero((df['site_type'] == site_type) & (np.sum(cleavage_counts, axis=1) > min_read_count))[0]
            cleavage_counts = cleavage_counts[filter_index, :]

            cleavage_distribs = cleavage_counts / np.sum(cleavage_counts, axis=1).reshape(-1, 1)

            n_events[tissue_i] = cleavage_counts.shape[0]
            n_reads[tissue_i] = np.sum(cleavage_counts) / n_events[tissue_i]

            cleavage_distrib = np.mean(cleavage_distribs, axis=0)

            if plot_tissue_specific :
                l1, = plt.plot(np.arange(186)[min_pos:max_pos]-pas_offset, cleavage_distrib[min_pos:max_pos], label=tissue, alpha=0.75, linewidth=2)
                ls.append(l1)
            
            cum_cleavage_distribs[:, tissue_i] = cleavage_distrib
        
        
        
        cum_cleavage_distrib = np.mean(cum_cleavage_distribs, axis=1)
        
        if plot_pooled :
            l1, = plt.plot(np.arange(186)[min_pos:max_pos]-pas_offset, cum_cleavage_distrib[min_pos:max_pos], c='black', label='Pooled', linewidth=3)
            ls.append(l1)

        avg_n_events = round(np.mean(n_events), 1)
        avg_n_reads = round(np.mean(n_reads), 1)

        plt.title(site_type + '(#Events/Tissue = ' + str(avg_n_events) + ', #Reads/Event = ' + str(avg_n_reads) + ')', fontsize=18)
        plt.xlabel('Position relative to PAS Start', fontsize=16)
        plt.ylabel('Cleavage fraction', fontsize=16)
        plt.tick_params(axis='both', which='minor', labelsize=16)
        plt.tick_params(axis='both', which='major', labelsize=16)

        #plt.legend(handles=ls, fontsize=14)
        plt.tight_layout()
        plt.show()

#Calculate predicted vs. observed mean cut position per PAS sequence
def get_avgpos_pred_vs_true(df, leslie_cleavage_count_dict, pred_cleavage_prob, tissue_index, site_types, cut_start=60, cut_end=105, count_filter=100, pooled=True) :

    pooled_cleavage_count_matrix = np.zeros((len(tissue_index), leslie_cleavage_count_dict['hek293'].shape[0], cut_end - cut_start))

    for tissue_i in range(0, len(tissue_index)) :
        tissue = tissue_index[tissue_i]
        cleavage_counts = np.array(leslie_cleavage_count_dict[tissue].todense())
        pooled_cleavage_count_matrix[tissue_i, :, :] = cleavage_counts[:, cut_start:cut_end]

    if pooled :
        pooled_cleavage_count_matrix = np.sum(pooled_cleavage_count_matrix, axis=0)
    else :
        pooled_cleavage_count_matrix = np.mean(pooled_cleavage_count_matrix, axis=0)
    
    total_cuts_true = np.ravel(np.sum(pooled_cleavage_count_matrix, axis=1))

    filter_index = np.nonzero((df['site_type'].isin(site_types)) & (total_cuts_true > count_filter))[0]
    pooled_cleavage_count_matrix = pooled_cleavage_count_matrix[filter_index, :]
    cuts_pred = np.array(pred_cleavage_prob.todense())[filter_index, cut_start:cut_end]

    cuts_true = pooled_cleavage_count_matrix / np.sum(pooled_cleavage_count_matrix, axis=1).reshape(-1, 1)
    cuts_pred = cuts_pred / np.sum(cuts_pred, axis=1).reshape(-1, 1)

    avgpos_true = np.ravel(np.sum(cuts_true * np.arange(cuts_true.shape[1]).reshape(1, -1), axis=1))
    avgpos_pred = np.ravel(np.sum(cuts_pred * np.arange(cuts_pred.shape[1]).reshape(1, -1), axis=1))

    return avgpos_pred, avgpos_true

#Plot avgpos pred vs true per tissue type
def plot_pred_vs_observed_mean_cut(df, leslie_cleavage_count_dict, pred_cleavage_prob, leslie_tissue_index) :
    leslie_tissue_names = []
    leslie_blacklist = {'mcf10a1' : True, 'mcf10a_hras1' : True, 'bcells1' : True}

    for tissue in leslie_tissue_index :
        if tissue not in leslie_blacklist :
            leslie_tissue_names.append(tissue)

    n_rows = 3
    n_cols = 5

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    site_types = ['UTR3']
    count_filter = 200
    pooled=False

    avgcut_r2_vec = np.zeros(len(leslie_tissue_names))

    for row_i in range(0, n_rows) :
        for col_i in range(0, n_cols) :
            ax[row_i, col_i].axis('off')

    for tissue_i in range(0, len(leslie_tissue_names)) :
        tissue = leslie_tissue_names[tissue_i]
        
        if tissue == 'pooled' :
            continue

        row_i = int(tissue_i / n_cols)
        col_i = int(tissue_i % n_cols)

        ax[row_i, col_i].axis('on')

        avgpos_pred, avgpos_true = get_avgpos_pred_vs_true(df, leslie_cleavage_count_dict, pred_cleavage_prob, np.array([tissue], dtype=np.object), site_types, count_filter=count_filter, pooled=pooled)
        
        ax[row_i, col_i].scatter(avgpos_pred, avgpos_true, s=5, c='black', alpha=0.15)
        
        r_val, _ = pearsonr(avgpos_pred, avgpos_true)
        
        plt.sca(ax[row_i, col_i])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Predicted cut position', fontsize=16)
        plt.ylabel('Observed cut position', fontsize=16)
        
        plt.title(tissue + ' (R^2 = ' + str(round(r_val * r_val, 2)) + ')')
        
        avgcut_r2_vec[tissue_i] = round(r_val * r_val, 2)

    col_i +=1
    ax[row_i, col_i].axis('on')

    pooled_filter = 500
    avgpos_pred, avgpos_true = get_avgpos_pred_vs_true(df, leslie_cleavage_count_dict, pred_cleavage_prob, leslie_tissue_index, site_types, count_filter=pooled_filter, pooled=pooled)

    ax[row_i, col_i].scatter(avgpos_pred, avgpos_true, s=5, c='darkblue', alpha=0.15)

    r_val, _ = pearsonr(avgpos_pred, avgpos_true)

    plt.sca(ax[row_i, col_i])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Predicted cut', fontsize=14)
    plt.ylabel('Observed cut', fontsize=14)

    plt.title('Pooled (R^2 = ' + str(round(r_val * r_val, 2)) + ')', fontsize=14)

    plt.tight_layout()
    plt.show()

#Plot avgpos pred vs true per tissue type, as barchart
def plot_pred_vs_observed_mean_cut_bar(df, leslie_cleavage_count_dict, pred_cleavage_prob, leslie_tissue_index) :
    leslie_tissue_names = []
    leslie_blacklist = {'mcf10a1' : True, 'mcf10a_hras1' : True, 'bcells1' : True}

    for tissue in leslie_tissue_index :
        if tissue not in leslie_blacklist :
            leslie_tissue_names.append(tissue)
    
    site_types = ['UTR3']
    count_filter = 200
    pooled=False

    avgcut_r2_vec = np.zeros(len(leslie_tissue_names))

    for tissue_i in range(0, len(leslie_tissue_names)) :
        tissue = leslie_tissue_names[tissue_i]
        
        if tissue == 'pooled' :
            continue

        avgpos_pred, avgpos_true = get_avgpos_pred_vs_true(df, leslie_cleavage_count_dict, pred_cleavage_prob, np.array([tissue], dtype=np.object), site_types, count_filter=count_filter, pooled=pooled)
        
        r_val, _ = pearsonr(avgpos_pred, avgpos_true)
        avgcut_r2_vec[tissue_i] = round(r_val * r_val, 2)

    #Plot avgcut r2
    f = plt.figure(figsize=(3, 4))

    plt.barh(np.arange(len(leslie_tissue_names)), avgcut_r2_vec[::-1], edgecolor='black', linewidth=1, alpha=1.0)

    plt.yticks(np.arange(len(leslie_tissue_names)), leslie_tissue_names[::-1], fontsize=14)

    plt.xlim(0, np.max(avgcut_r2_vec) + 0.05)
    plt.ylim(-0.75, len(leslie_tissue_names))

    plt.xticks([0, 0.3, 0.6], fontsize=14)
    plt.xlabel('R^2', fontsize=14)

    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().xaxis.tick_top()

    plt.tight_layout()
    plt.show()


#Regression and cross-validation helper functions
def safe_kl_log(num, denom) :
    log_vec = np.zeros(num.shape)
    log_vec[(num > 0) & (denom > 0)] = np.log(num[(num > 0) & (denom > 0)] / denom[(num > 0) & (denom > 0)])
    
    return log_vec

def k_fold_optimize_linear(k, X, y, l2_lambdas=None, l1_lambdas=None, debias_l1=False, min_params=1, max_params=1000) :
    min_loss = np.inf
    min_param = 0
    
    if l2_lambdas is not None :
        for l2_lambda in l2_lambdas :
            total_loss, _ = k_fold_cross_linear(k, X, y, l2_lambda=l2_lambda, l1_lambda=None)
            if total_loss < min_loss :
                min_loss = total_loss
                min_param = l2_lambda
    elif l1_lambdas is not None :
        for l1_lambda in l1_lambdas :
            _, _, _, _, w_bundle = fit_linear_model(X, y, X, y, l2_lambda=None, l1_lambda=l1_lambda)
            w, _ = w_bundle
            n_nonzero = len(np.nonzero(w)[0])
            total_loss = np.inf
            if n_nonzero > 0 :
                if not debias_l1 :
                    total_loss, _ = k_fold_cross_linear(k, X, y, l2_lambda=None, l1_lambda=l1_lambda)
                else :
                    total_loss, _ = k_fold_cross_linear(k, X[:, w != 0], y, l2_lambda=None, l1_lambda=None)

            if total_loss < min_loss and n_nonzero >= min_params and n_nonzero <= max_params :
                min_loss = total_loss
                min_param = l1_lambda
            
            print(str(l1_lambda) + ' = ' + str(total_loss))
    
    return min_loss, min_param

def k_fold_optimize_logistic(k, X, y, l2_lambdas=None) :
    min_loss = np.inf
    min_param = 0
    
    if l2_lambdas is not None :
        for l2_lambda in l2_lambdas :
            total_loss, _ = k_fold_cross_logistic(k, X, y, l2_lambda=l2_lambda, l1_lambda=None)
            if total_loss < min_loss :
                min_loss = total_loss
                min_param = l2_lambda
    
    return min_loss, min_param


def k_fold_cross_linear(k, X, y, l2_lambda=None, l1_lambda=None) :
    n_batches = k
    batch_size = int(X.shape[0] / float(k))
    if batch_size * k < X.shape[0] - 1 :
        n_batches += 1
    
    total_loss = 0
    total_y_hat = []
    
    for batch_index in range(n_batches) :
        X_train = None
        y_train = None
        X_test = None
        y_test = None
        if batch_index < n_batches - 1 :
            
            if batch_index == 0 :
                X_train = X[(batch_index + 1) * batch_size:, :]
                y_train = y[(batch_index + 1) * batch_size:]
            elif isinstance(X, sp.csr_matrix) or isinstance(X, sp.csc_matrix) :
                X_train = sp.vstack([X[:(batch_index - 1) * batch_size, :], X[(batch_index + 1) * batch_size:, :]])
                y_train = np.concatenate([y[:(batch_index - 1) * batch_size], y[(batch_index + 1) * batch_size:]])
            else :
                X_train = np.concatenate([X[:(batch_index - 1) * batch_size, :], X[(batch_index + 1) * batch_size:, :]], axis=0)
                y_train = np.concatenate([y[:(batch_index - 1) * batch_size], y[(batch_index + 1) * batch_size:]])
            
            X_test = X[batch_index * batch_size: (batch_index + 1) * batch_size, :]
            y_test = y[batch_index * batch_size: (batch_index + 1) * batch_size]
        else :
            
            X_train = X[:batch_index * batch_size, :]
            y_train = y[:batch_index * batch_size]
            
            X_test = X[batch_index * batch_size:, :]
            y_test = y[batch_index * batch_size:]
        
        y_test_hat, sse, _, _, _ = fit_linear_model(X_train, y_train, X_test, y_test, l2_lambda, l1_lambda)
        
        total_loss += sse
        total_y_hat.append(y_test_hat)
    
    total_y_hat = np.concatenate(total_y_hat, axis=0)
    
    return total_loss, total_y_hat

def k_fold_cross_logistic(k, X, y, l2_lambda=None) :
    n_batches = k
    batch_size = int(X.shape[0] / float(k))
    if batch_size * k < X.shape[0] - 1 :
        n_batches += 1
    
    total_loss = 0
    total_y_hat = []
    
    for batch_index in range(n_batches) :
        X_train = None
        y_train = None
        X_test = None
        y_test = None
        if batch_index < n_batches - 1 :
            
            if batch_index == 0 :
                X_train = X[(batch_index + 1) * batch_size:, :]
                y_train = y[(batch_index + 1) * batch_size:]
            elif isinstance(X, sp.csr_matrix) or isinstance(X, sp.csc_matrix) :
                X_train = sp.vstack([X[:(batch_index - 1) * batch_size, :], X[(batch_index + 1) * batch_size:, :]])
                y_train = np.concatenate([y[:(batch_index - 1) * batch_size], y[(batch_index + 1) * batch_size:]])
            else :
                X_train = np.concatenate([X[:(batch_index - 1) * batch_size, :], X[(batch_index + 1) * batch_size:, :]], axis=0)
                y_train = np.concatenate([y[:(batch_index - 1) * batch_size], y[(batch_index + 1) * batch_size:]])
            
            X_test = X[batch_index * batch_size: (batch_index + 1) * batch_size, :]
            y_test = y[batch_index * batch_size: (batch_index + 1) * batch_size]
        else :
            
            X_train = X[:batch_index * batch_size, :]
            y_train = y[:batch_index * batch_size]
            
            X_test = X[batch_index * batch_size:, :]
            y_test = y[batch_index * batch_size:]
        
        y_test_hat, kldiv, _, _, _ = fit_logistic_model(X_train, y_train, X_test, y_test, l2_lambda)
        
        total_loss += kldiv
        total_y_hat.append(y_test_hat)
    
    total_y_hat = np.concatenate(total_y_hat, axis=0)
    
    return total_loss, total_y_hat
    

def fit_linear_model(X_train, y_train, X_test, y_test, l2_lambda=None, l1_lambda=None) :
    
    lr = sklinear.LinearRegression()
    if l2_lambda is not None :
        lr = sklinear.Ridge(alpha=l2_lambda)
    elif l1_lambda is not None :
        lr = sklinear.Lasso(alpha=l1_lambda)
    
    lr.fit(X_train, y_train)
    
    y_test_hat = lr.predict(X_test)
    
    SSE = (y_test - y_test_hat).T.dot(y_test - y_test_hat)
    y_test_average = np.average(y_test, axis=0)
    SStot = (y_test - y_test_average).T.dot(y_test - y_test_average)
    rsquare = 1.0 - (SSE / SStot)

    accuracy = float(np.count_nonzero(np.sign(y_test) == np.sign(y_test_hat))) / float(X_test.shape[0])
    
    return y_test_hat, SSE, rsquare, accuracy, (lr.coef_, lr.intercept_)

def predict_linear_model(X, w, w_0) :
    return X.dot(w) + w_0

def fit_logistic_model(X_train, y_train, X_test, y_test, l2_lambda=None) :
    
    f_loss = lambda w_bundle, X=X_train, y_true=y_train, alpha=l2_lambda: kl_div_loss(w_bundle, X, y_true, alpha)
    f_grad = lambda w_bundle, X=X_train, y_true=y_train, alpha=l2_lambda: kl_div_gradients(w_bundle, X, y_true, alpha)
    w_bundle_init = np.zeros(X_train.shape[1] + 1)
    res = minimize(f_loss, w_bundle_init, method='BFGS', jac=f_grad, options={'disp': False})

    w_bundle = res.x
    w_0 = w_bundle[0]
    w = w_bundle[1:]

    y_test_hat = get_y_pred(X_test, w, w_0)
    
    SSE = (y_test - y_test_hat).T.dot(y_test - y_test_hat)
    y_test_average = np.average(y_test, axis=0)
    SStot = (y_test - y_test_average).T.dot(y_test - y_test_average)
    rsquare = 1.0 - (SSE / SStot)

    accuracy = float(np.count_nonzero(np.sign(y_test) == np.sign(y_test_hat))) / float(X_test.shape[0])
    
    return y_test_hat, kl_div_loss(w_bundle, X_test, y_test, alpha=l2_lambda) * float(X_test.shape[0]), rsquare, accuracy, (w, w_0)

def predict_logistic_model(X, w, w_0) :
    return get_y_pred(X, w, w_0)

def fit_loocv_model(X, y, l2_lambda=None) :
    lr = sklinear.LinearRegression()
    if l2_lambda is not None :
        lr = sklinear.Ridge(alpha=l2_lambda)

    y_hat = cross_val_predict(lr, X, y, cv=X.shape[0])

    SSE = (y - y_hat).T.dot(y - y_hat)
    y_average = np.average(y, axis=0)
    SStot = (y - y_average).T.dot(y - y_average)
    rsquare = 1.0 - (SSE / SStot)

    accuracy = float(np.count_nonzero(np.sign(y) == np.sign(y_hat))) / float(X.shape[0])
    
    lr.fit(X, y)
    
    return y_hat, rsquare, accuracy, (lr.coef_, lr.intercept_)

def predict_with_model(X, y, weight_bundle) :
    lr = sklinear.LinearRegression()
    lr.coef_ = weight_bundle[0]
    lr.intercept_ = weight_bundle[1]
    
    y_hat = lr.predict(X)

    SSE = (y - y_hat).T.dot(y - y_hat)
    y_average = np.average(y, axis=0)
    SStot = (y - y_average).T.dot(y - y_average)
    rsquare = 1.0 - (SSE / SStot)

    accuracy = float(np.count_nonzero(np.sign(y) == np.sign(y_hat))) / float(X.shape[0])
    
    return y_hat, rsquare, accuracy

def get_y_pred(X, w, w_0) :
    score = X.dot(w) + w_0
    return 1. / (1. + np.exp(-score))

def kl_div_loss(w_bundle, X, y_true, alpha=0.0) :
    w_0 = w_bundle[0]
    w = w_bundle[1:]
    
    y_pred = get_y_pred(X, w, w_0)
    kl = y_true * safe_kl_log(y_true, y_pred) + (1. - y_true) * safe_kl_log((1. - y_true), (1. - y_pred)) + (1./2.) * alpha * np.dot(w, w)
    
    return np.mean(kl)

def kl_div_gradients(w_bundle, X, y_true, alpha=0.0) :
    w_0 = w_bundle[0]
    w = w_bundle[1:]
    N = float(X.shape[0])
    y_pred = get_y_pred(X, w, w_0)
    
    kl_grad_w = (1. / N) * X.T.dot(y_pred - y_true) + alpha * w
    kl_grad_w_0 = (1. / N) * np.sum(y_pred - y_true)
    
    w_bundle_grads = np.zeros(w.shape[0] + 1)
    w_bundle_grads[0] = kl_grad_w_0
    w_bundle_grads[1:] = kl_grad_w
    
    return w_bundle_grads

#Evaluate and plot prediction performance against cell types using Pooled-APADB fitted APARENT model
def evaluate_predicted_vs_observed_tissues(pair_df, source_data, tissue_index, suffix_index, site_nums, site_types, pseudo_count, max_n_members) :
    #Specify cell type-specific min count filters
    count_filters = {
        'apadb_pooled' : 1000,
        'apadb_kidney' : 20,
        'apadb_pancreas' : 10,
        'apadb_monocytes' : 20,
        'apadb_all' : 20,
        'apadb_pdac' : 20,
        'apadb_prcc' : 20,
        'apadb_full_blood' : 500,
        'apadb_hlf' : 500,
        
        'leslie_pooled' : 1000,
        'leslie_hek293' : 50,
        'leslie_mcf10a_hras2' : 50,
        'leslie_mcf10a1' : 50,
        'leslie_mcf10a2' : 50,
        'leslie_mcf10a_hras1' : 50,
        'leslie_bcells1' : 50,
        'leslie_mcf7' : 50,
        'leslie_bcells2' : 50,
        'leslie_ovary' : 50,
        'leslie_breast' : 50,
        'leslie_brain' : 50,
        'leslie_skmuscle' : 50,
        'leslie_blcl' : 50,
        'leslie_hES' : 50,
        'leslie_testis' : 50,
        'leslie_hela' : 50,
        'leslie_ntera' : 50
    }

    max_cut_region = 60
    min_distance = 40
    max_distance = 4000

    #Apply global filters
    df_to_use = pair_df.loc[pair_df.site_type_prox.isin(site_types)]
    df_to_use = df_to_use.loc[df_to_use.site_type_dist.isin(site_types)]
    if site_nums is not None :
        df_to_use = df_to_use.loc[df_to_use.sitenum_prox.isin(site_nums)]

    df_to_use = df_to_use.query("(cut_end_prox - cut_start_prox <= " + str(max_cut_region) + ") and (cut_end_dist - cut_start_dist <= " + str(max_cut_region) + ")")
    df_to_use = df_to_use.query("(distance >= " + str(min_distance) + ") and (distance <= " + str(max_distance) + ")")

    df_to_use = df_to_use.query("mirna_prox == mirna_dist")

    n_rows = len(tissue_index)
    n_cols = 6

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))

    r2_tissues = np.zeros(len(tissue_index))
    auc_tissues = np.zeros(len(tissue_index))
    n_samples_tissues = np.zeros(len(tissue_index))
    read_depth_tissues = np.zeros(len(tissue_index))
    roc_tissues = []

    for tissue_i in range(0, len(tissue_index)) :
        tissue = tissue_index[tissue_i]
        source_data = source_data
        special_mode = suffix_index[tissue_i]
        
        tissue_name = tissue.replace('pooled', source_data + ' pooled')
        
        mode_list = [
            [0, False, False],
            [3, True, True]
        ]
        for [col_offset, differentials_only, use_logodds] in mode_list :
            count_col = source_data + '_count' + special_mode + '_' + tissue + '_prox'
            total_count_col = source_data + '_pair_count' + special_mode + '_' + tissue
            curr_df = df_to_use.query(total_count_col + " >= " + str(count_filters[source_data + '_' + tissue]))
            if differentials_only :
                curr_df = curr_df.query(count_col + " != " + total_count_col + " and " + count_col + " != 0")

            total_count = np.ravel(curr_df[total_count_col].values)

            true_ratio = np.ravel(((curr_df[count_col] + pseudo_count) / (curr_df[total_count_col] + 2. * pseudo_count)).values)
            true_logodds = np.log(true_ratio / (1. - true_ratio))

            pred_ratio = np.ravel(curr_df['iso_pred'].values)
            pred_logodds = np.log(pred_ratio / (1. - pred_ratio))

            true_metric = np.ravel(true_ratio)
            pred_metric = np.ravel(pred_ratio)
            if use_logodds :
                true_metric = np.ravel(true_logodds)
                pred_metric = np.ravel(pred_logodds)

            if max_n_members is not None :
                sort_index = np.argsort(total_count)[::-1]
                total_count = total_count[sort_index[:max_n_members]]
                true_ratio = true_ratio[sort_index[:max_n_members]]
                pred_ratio = pred_ratio[sort_index[:max_n_members]]
                true_metric = true_metric[sort_index[:max_n_members]]
                pred_metric = pred_metric[sort_index[:max_n_members]]

            r_val, _ = pearsonr(pred_metric, true_metric)

            n_samples = len(total_count)
            avg_reads = np.mean(total_count)

            annot_text = tissue_name + '\n'
            annot_text += 'samples = ' + str(n_samples) + '\n'
            annot_text += 'depth = ' + str(round(avg_reads, 2)) + '\n'
            if not use_logodds :
                annot_text += 'Isoform Proportions'
            else :
                annot_text += 'Isoform Log Odds'
            ax[tissue_i, 0 + col_offset].text(0.01, 0.5, annot_text, verticalalignment='center', horizontalalignment='left', transform=ax[tissue_i, 0 + col_offset].transAxes, fontsize=10)
            plt.sca(ax[tissue_i, 0 + col_offset])
            plt.axis('off')
            plt.xticks([], [])
            plt.yticks([], [])


            true_label = np.zeros(true_ratio.shape[0])
            true_label[true_ratio <= 0.5] = 0
            true_label[true_ratio > 0.5] = 1

            fpr, tpr, _ = roc_curve(true_label, pred_ratio)
            auc = roc_auc_score(true_label, pred_ratio)

            fpr_highconf, tpr_highconf, _ = roc_curve(true_label[(true_ratio <= 0.25) | (true_ratio >= 0.75)], pred_ratio[(true_ratio <= 0.25) | (true_ratio >= 0.75)])
            auc_highconf = roc_auc_score(true_label[(true_ratio <= 0.25) | (true_ratio >= 0.75)], pred_ratio[(true_ratio <= 0.25) | (true_ratio >= 0.75)])

            ax[tissue_i, 1 + col_offset].scatter(pred_metric, true_metric, alpha=0.25, c='black', s=5)
            plt.sca(ax[tissue_i, 1 + col_offset])
            if differentials_only :
                plt.ylabel('Differentials only')
            else :
                plt.ylabel(tissue_name)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title('R^2 = ' + str(round(r_val * r_val, 2)))

            l1 = ax[tissue_i, 2 + col_offset].plot(fpr, tpr, linewidth=2, color='black', label='AUC = ' + str(round(auc, 3)))
            l1_highconf = ax[tissue_i, 2 + col_offset].plot(fpr_highconf, tpr_highconf, linewidth=2, color='red', label='AUC = ' + str(round(auc_highconf, 3)))

            plt.sca(ax[tissue_i, 2 + col_offset])
            plt.xticks([], [])
            plt.yticks([], [])
            plt.xlim(0-0.01, 1)
            plt.ylim(0, 1+0.01)
            plt.legend(handles=[l1[0], l1_highconf[0]], fontsize=9, frameon=False, loc='lower right')
            
            if not differentials_only :
                r2_tissues[tissue_i] = round(r_val * r_val, 2)
                auc_tissues[tissue_i] = round(auc, 3)
                n_samples_tissues[tissue_i] = len(total_count)
                read_depth_tissues[tissue_i] = round(np.mean(total_count), 1)
                roc_tissues.append([fpr, tpr])

    plt.show()

    return r2_tissues, auc_tissues, n_samples_tissues, read_depth_tissues, roc_tissues

def plot_performance_double_bar(tissue_index, measures_1, measure_2) :
    f, ax = plt.subplots(1, 2, figsize=(8, 6))

    #Plot Number of Sample APA sites per cell type
    ax[0].barh(np.arange(len(tissue_index)), measures_1[::-1], edgecolor='black', linewidth=1, alpha=1.0)
    plt.sca(ax[0])
    plt.yticks(np.arange(len(tissue_index)), tissue_index[::-1], fontsize=14)
    plt.ylim(-0.75, len(tissue_index))
    plt.gca().xaxis.tick_top()

    plt.xlabel('Num APA Sites', fontsize=14)

    #Plot Mean Read Depth per cell type
    ax[1].barh(np.arange(len(tissue_index)), measure_2[::-1], edgecolor='black', linewidth=1, alpha=1.0)
    plt.sca(ax[1])
    plt.yticks(np.arange(len(tissue_index)), tissue_index[::-1], fontsize=14)
    plt.ylim(-0.75, len(tissue_index))
    plt.gca().xaxis.tick_top()

    plt.xlabel('Avg Read Depth', fontsize=14)

    plt.tight_layout()
    plt.show()

