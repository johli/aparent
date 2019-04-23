import pandas as pd
import numpy as np
import scipy.sparse as sp

#Random MPRA sorting and shuffling functions

class LibraryPreparer :
    
    def __init__(self, preparer_type_id) :
        self.preparer_type_id = preparer_type_id
    
    def _prepare(self, library_dict) :
        raise NotImplementedError()

class LibraryCountFilter(LibraryPreparer) :
    
    def __init__(self, minimum_count) :
        super(LibraryCountFilter, self).__init__('lib_count_filter')
        
        self.minimum_count = minimum_count
    
    def _prepare(self, library_dict) :
        keep_index = np.nonzero(library_dict['data']['total_count'] >= self.minimum_count)[0]
        
        new_library_dict = { 'metadata' : library_dict['metadata'] }
        new_library_dict['data'] = library_dict['data'].iloc[keep_index].reset_index(drop=True)
        new_library_dict['cuts'] = library_dict['cuts'][keep_index]
        
        return new_library_dict
    
    def __call__(self, library_dict) :
        return self._prepare(library_dict)

class SubLibraryCountFilter(LibraryPreparer) :
    
    def __init__(self, min_count_dict) :
        super(SubLibraryCountFilter, self).__init__('sublib_count_filter')
        
        self.min_count_dict = min_count_dict
    
    def _prepare(self, library_dict) :
        keep_index = []
        i = 0
        for _, row in library_dict['data'].iterrows() :
            if i % 100000 == 0 :
                print("Filtering sequence " + str(i))
            
            if row['library_index'] not in self.min_count_dict :
                keep_index.append(i)
            elif row['total_count'] >= self.min_count_dict[row['library_index']] :
                keep_index.append(i)
            i += 1
        
        new_library_dict = { 'metadata' : library_dict['metadata'] }
        new_library_dict['data'] = library_dict['data'].iloc[keep_index].reset_index(drop=True)
        new_library_dict['cuts'] = library_dict['cuts'][keep_index]
        
        return new_library_dict
    
    def __call__(self, library_dict) :
        return self._prepare(library_dict)

class LibrarySelector(LibraryPreparer) :
    
    def __init__(self, included_libs) :
        super(LibrarySelector, self).__init__('lib_selector')
        
        self.included_libs = included_libs
    
    def _prepare(self, library_dict) :
        keep_index = np.nonzero(library_dict['data']['library_index'].isin(self.included_libs))[0]
        
        new_library_dict = { 'metadata' : library_dict['metadata'] }
        new_library_dict['data'] = library_dict['data'].iloc[keep_index].reset_index(drop=True)
        new_library_dict['cuts'] = library_dict['cuts'][keep_index]
        
        return new_library_dict
    
    def __call__(self, library_dict) :
        return self._prepare(library_dict)

class LibraryBalancer(LibraryPreparer) :
    
    def __init__(self, included_libs) :
        super(LibraryBalancer, self).__init__('lib_balancer')
        
        self.included_libs = included_libs
    
    def _prepare(self, library_dict) :
        L_included = self.included_libs
        
        arranged_index_len = 0

        arranged_index_len = int(np.sum([len(np.nonzero(library_dict['data']['library_index'] == lib)[0]) for lib in L_included]))
        min_join_len = int(np.min([len(np.nonzero(library_dict['data']['library_index'] == lib)[0]) for lib in L_included]))

        arranged_index = np.zeros(arranged_index_len, dtype=np.int)

        arranged_remainder_index = 0
        arranged_join_index = arranged_index_len - len(L_included) * min_join_len

        for lib_i in range(0, len(L_included)) :
            lib = L_included[lib_i]

            print('Arranging lib ' + str(lib))

            #1. Get indexes of each Library
            lib_index = np.nonzero(library_dict['data']['library_index'] == lib)[0]

            #2. Sort indexes of each library by count
            lib_count = library_dict['data'].iloc[lib_index]['total_count']
            sort_index_lib = np.argsort(lib_count)
            lib_index = lib_index[sort_index_lib]

            #3. Shuffle indexes of each library modulo 2
            even_index_lib = np.nonzero(np.arange(len(lib_index)) % 2 == 0)[0]
            odd_index_lib = np.nonzero(np.arange(len(lib_index)) % 2 == 1)[0]

            lib_index_even = lib_index[even_index_lib]
            lib_index_odd = lib_index[odd_index_lib]

            lib_index = np.concatenate([lib_index_even, lib_index_odd])

            #4. Join modulo 2
            i = 0
            for j in range(len(lib_index) - min_join_len, len(lib_index)) :
                arranged_index[arranged_join_index + i * len(L_included) + lib_i] = lib_index[j]
                i += 1

            #5. Append remainder
            for j in range(0, len(lib_index) - min_join_len) :
                arranged_index[arranged_remainder_index] = lib_index[j]
                arranged_remainder_index += 1

        new_library_dict = { 'metadata' : library_dict['metadata'] }
        new_library_dict['data'] = library_dict['data'].iloc[arranged_index].reset_index(drop=True)
        new_library_dict['cuts'] = library_dict['cuts'][arranged_index]

        #Perform final read count control check between dataframe and cut matrix
        total_count_from_cuts = np.ravel(new_library_dict['cuts'].sum(axis=1)) + np.ravel(new_library_dict['data']['distal_count'].values)
        if not np.all(total_count_from_cuts == np.array(new_library_dict['data']['total_count'].values)) :
            print('Error! Count mismatch between dataframe and cut matrix.')
        
        return new_library_dict
    
    def __call__(self, library_dict) :
        return self._prepare(library_dict)

def plot_cumulative_library_proportion(library_dict, percentile_step=0.05, figsize=(12, 8), n_xticks=10, n_yticks=10) :
    
    library_fractions_from_top = np.linspace(0, 1, num=int(1. / percentile_step) + 1)[1:]
    libs = library_dict['data']['library'].unique()

    cum_fraction = np.zeros((len(library_fractions_from_top), len(libs)))

    total_lib_size = float(len(library_dict['data']))

    frac_i = 0
    for library_fraction in library_fractions_from_top :

        lib_i = 0
        for lib in libs :
            lib_slice = library_dict['data'].iloc[-int(library_fraction * total_lib_size):]

            lib_size = len(np.nonzero((lib_slice['library'] == lib))[0])

            curr_frac = float(lib_size) / float(len(lib_slice))

            cum_fraction[frac_i, lib_i] = curr_frac

            lib_i += 1

        frac_i += 1
    
    fig = plt.subplots(figsize=figsize)

    plt.stackplot(library_fractions_from_top, np.fliplr(cum_fraction.T), labels=libs)
    plt.legend(loc='upper left', fontsize=12)

    plt.xticks(np.linspace(0, 1, num=n_xticks + 1)[:-1], np.round(np.linspace(0, 1, num=n_xticks + 1), 2)[:-1], fontsize=14, rotation=45)
    plt.yticks(np.linspace(0, 1, num=n_yticks + 1), np.round(np.linspace(0, 1, num=n_yticks + 1), 2), fontsize=14)

    plt.xlim(np.min(library_fractions_from_top), np.max(library_fractions_from_top))
    plt.ylim(0, 1)

    plt.xlabel('Percentile of data (low to high read count)', fontsize=14)
    plt.ylabel('Library proportion of Percentile to 100%', fontsize=14)

    plt.title('Cumulative library proportion', fontsize=16)

    plt.tight_layout()
    plt.show()

def plot_library_cut_profile(library_dict, figsize=(12, 8)) :

    f = plt.figure(figsize=figsize)

    libs = library_dict['data']['library'].unique()

    ls = []
    for lib in libs :

        lib_index = np.nonzero((library_dict['data']['library'] == lib))[0]
        proximal_profile = np.ravel(library_dict['cuts'][lib_index].sum(axis=0))
        proximal_profile /= np.sum(proximal_profile)

        la, = plt.plot(np.arange(len(proximal_profile)), proximal_profile, linewidth=2, label=lib)
        ls.append(la)


    #Proximal 1
    plt.axvline(x=70, linewidth=2, c='black', linestyle='--')
    plt.axvline(x=70 + 6, linewidth=2, c='black', linestyle='--')
    plt.axvline(x=70 + 21, linewidth=2, c='orange', linestyle='--')

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Position', fontsize=14)
    plt.ylabel('Read count', fontsize=14)
    plt.title('Proximal site', fontsize=16)
    plt.tight_layout()

    plt.legend(handles = ls, fontsize=12)

    plt.show()


def plot_individual_library_count_distribution(library_dict, figsize=(12, 8), n_xticks=10, y_max=500) :

    total_count = np.ravel(library_dict['data']['total_count'].values)
    libs = library_dict['data']['library'].unique()

    fig = plt.figure(figsize=figsize)
    ls = []

    for lib in libs :
        lib_index = np.nonzero(library_dict['data']['library'] == lib)[0]
        lib_slice = library_dict['data'].iloc[lib_index]

        lib_count = np.ravel(lib_slice['total_count'].values)
        lib_frac = np.arange(len(lib_slice)) / float(len(lib_slice))

        lt, = plt.plot(lib_frac, lib_count, linewidth=2, label=lib)
        ls.append(lt)

    plt.legend(handles=ls, loc='upper left', fontsize=12)

    plt.xticks(np.round(np.linspace(0, 1, num=n_xticks + 1), 2), np.round(np.linspace(0, 1, num=n_xticks + 1), 2), fontsize=14, rotation=45)
    plt.yticks(fontsize=14)

    plt.xlim(0, 1)
    plt.ylim(0, y_max)

    plt.xlabel('Percentile of data', fontsize=14)
    plt.ylabel('Read count', fontsize=14)

    plt.title('Individual library count distribution', fontsize=16)

    plt.tight_layout()
    plt.show()

def plot_combined_library_count_distribution(library_dict, figsize=(12, 8), n_xticks=10, x_min=0, x_max=1, y_max=500) :

    total_count = np.ravel(library_dict['data']['total_count'].values)
    total_lib_frac = np.arange(total_count.shape[0]) / float(total_count.shape[0])
    libs = library_dict['data']['library'].unique()

    fig = plt.figure(figsize=figsize)
    ls = []

    for lib in libs :
        lib_index = np.nonzero(library_dict['data']['library'] == lib)[0]
        lib_slice = library_dict['data'].iloc[lib_index]

        lib_count = np.ravel(lib_slice['total_count'].values)
        lib_frac = total_lib_frac[lib_index]

        lt, = plt.plot(lib_frac, lib_count, linewidth=2, label=lib)
        ls.append(lt)

    plt.legend(handles=ls, loc='upper left', fontsize=12)

    plt.xticks(np.round(np.linspace(0, 1, num=n_xticks + 1), 2), np.round(np.linspace(0, 1, num=n_xticks + 1), 2), fontsize=14, rotation=45)
    plt.yticks(fontsize=14)

    plt.xlim(x_min, x_max)
    plt.ylim(0, y_max)

    plt.xlabel('Percentile of data', fontsize=14)
    plt.ylabel('Read count', fontsize=14)

    plt.title('Combined library count distribution', fontsize=16)

    plt.tight_layout()
    plt.show()

#Designed MPRA aggregate functions

def group_dataframe(df, cuts, cut_start=None, min_total_count=1, drop_nans=False, nan_prox_range=[57, 87], misprime_filters=None, groupby_list=['master_seq']) :
    
    print('Collapsing with groupby = ' + str(groupby_list))
    
    df_copy = df.copy().reset_index(drop=True)
    
    cuts_dense = np.array(cuts.todense())
    if cut_start is not None :
        cuts_dense = np.array(cuts.todense())[:, cut_start:]
    
    cuts_dense = np.hstack([cuts_dense, np.ravel(df_copy['distal_count'].values).reshape(-1, 1)])
    
    df_copy['cuts'] = cuts_dense.tolist()
    
    print('Filtering...')
    print('Size before filtering = ' + str(len(df_copy)))
    
    
    df_copy['prox_ratio_temp'] = np.sum(cuts_dense[:, nan_prox_range[0]:nan_prox_range[1]], axis=1) / np.sum(cuts_dense, axis=1)
    
    if drop_nans :
        df_copy = df_copy.query("total_count >= " + str(min_total_count) + " and prox_ratio_temp > 0.0 and prox_ratio_temp < 1.0").reset_index(drop=True)
    else :
        df_copy = df_copy.query("total_count >= " + str(min_total_count)).reset_index(drop=True)
    
    if misprime_filters is not None :
        misprime_query = ""
        experiment_i = 0
        for experiment in misprime_filters :
            if experiment != 'all' :
                misprime_query += "not (experiment == '" + experiment + "' and ("
            else :
                misprime_query += "not ("
            for filter_i, filter_id in enumerate(misprime_filters[experiment]) :
                misprime_query += filter_id + " == True"
                if filter_i < len(misprime_filters[experiment]) - 1 :
                    misprime_query += " or "
                else :
                    misprime_query += ")"
            if experiment_i < len(misprime_filters) - 1 :
                misprime_query += ") and "
            else :
                misprime_query += ")"

            experiment_i += 1

        df_copy = df_copy.query(misprime_query).reset_index(drop=True)
    
    print('Size after filtering = ' + str(len(df_copy)))
    
    df_group = df_copy.groupby(groupby_list)
    
    list_f = lambda x: tuple(x)
    
    agg_dict = {
        'experiment' : 'first',
        'subexperiment' : 'first',
        'gene' : 'first',
        'significance' : 'first',
        'clinvar_id' : 'first',
        'variant' : 'first',
        'in_acmg' : 'first',
        'sitetype' : 'first',
        'wt_seq' : 'first',
        'predicted_logodds' : 'first',
        'predicted_usage' : 'first',
        'barcode' : list_f,
        'proximal_count' : list_f,
        'distal_count' : list_f,
        'total_count' : list_f,
        'cuts' : lambda x: tuple([np.array(l) for l in x])#tuple(x.tolist())
    }
    if 'master_seq' not in groupby_list :
        agg_dict['master_seq'] = 'first'
    
    df_agg = df_group.agg(agg_dict)
    
    print('Grouped dataframe.')
    
    return df_agg



def summarize_dataframe(df, min_barcodes=1, min_pooled_count=1, min_mean_count=1, prox_cut_start=55, prox_cut_end=85, isoform_pseudo_count=0, pooled_isoform_pseudo_count=0, cut_pseudo_count=0, drop_nans=False) :
    
    print('Filtering...')
    
    df['n_barcodes'] = df['barcode'].apply(lambda t: len(t))
    df['pooled_total_count'] = df['total_count'].apply(lambda t: np.sum(np.array(list(t))))
    df['mean_total_count'] = df['total_count'].apply(lambda t: np.mean(np.array(list(t))))
    
    df = df.query("n_barcodes >= " + str(min_barcodes) + " and pooled_total_count >= " + str(min_pooled_count) + " and mean_total_count >= " + str(min_mean_count)).copy()
    
    print('Summarizing...')
    
    df['pooled_cuts'] = df['cuts'].apply(lambda t: np.sum(np.array(list(t)), axis=0))
    df['mean_cuts'] = df['cuts'].apply(lambda t: np.mean(np.vstack([ x for x in list(t) ]), axis=0))
    
    
    df['pooled_cut_prob'] = df['cuts'].apply(lambda t: np.sum(np.array(list(t)), axis=0) / np.sum(np.array(list(t))))
    df['mean_cut_prob'] = df['cuts'].apply( lambda t: np.mean(np.vstack([ x / np.sum(x) for x in list(t) ]), axis=0) )
    
    df['proximal_count'] = df['cuts'].apply(lambda t: tuple([np.sum(x[prox_cut_start: prox_cut_end]) for x in t]))
    
    proximal_distrib = df['cuts'].apply(lambda t: tuple([(x[prox_cut_start: prox_cut_end] + cut_pseudo_count) / np.sum(x[prox_cut_start: prox_cut_end] + cut_pseudo_count) for x in t]))
    df['proximal_avgcut'] = proximal_distrib.apply(lambda t: tuple([np.sum(x * (np.arange(prox_cut_end - prox_cut_start))) for x in t]))
    
    df['pooled_proximal_count'] = df['proximal_count'].apply(lambda t: np.sum(np.array(list(t))))
    df['pooled_distal_count'] = df['distal_count'].apply(lambda t: np.sum(np.array(list(t))))
    
    df['proximal_usage'] = df.apply(lambda row: tuple([(p + isoform_pseudo_count) / (t + 2. * isoform_pseudo_count) for p, t in zip(list(row['proximal_count']), list(row['total_count']))]), axis=1)
    df['proximal_logodds'] = df['proximal_usage'].apply(lambda t: tuple([np.log(p / (1. - p)) for p in list(t)]))
    
    df['pooled_proximal_usage'] = (df['pooled_proximal_count'] + pooled_isoform_pseudo_count) / (df['pooled_total_count'] + 2. * pooled_isoform_pseudo_count)
    df['mean_proximal_usage'] = df['proximal_usage'].apply(lambda t: np.mean(list(t)))
    df['median_proximal_usage'] = df['proximal_usage'].apply(lambda t: np.median(list(t)))
    df['std_proximal_usage'] = df['proximal_usage'].apply(lambda t: np.std(list(t)))
    
    df['pooled_proximal_logodds'] = np.log(df['pooled_proximal_usage'] / (1. - df['pooled_proximal_usage']))
    df['mean_proximal_logodds'] = df['proximal_logodds'].apply(lambda t: np.mean(list(t)))
    df['median_proximal_logodds'] = df['proximal_logodds'].apply(lambda t: np.median(list(t)))
    df['std_proximal_logodds'] = df['proximal_logodds'].apply(lambda t: np.std(list(t)))
    
    df['mean_proximal_avgcut'] = df['proximal_avgcut'].apply(lambda t: np.mean(list(t)))
    df['median_proximal_avgcut'] = df['proximal_avgcut'].apply(lambda t: np.median(list(t)))
    df['std_proximal_avgcut'] = df['proximal_avgcut'].apply(lambda t: np.std(list(t)))
    
    
    #Proximal Vs. Distal
    df['competing_count'] = df['cuts'].apply(lambda t: tuple([np.sum(x[:prox_cut_start]) for x in t]))
    df['pooled_competing_count'] = df['competing_count'].apply(lambda t: np.sum(np.array(list(t))))
    
    
    df['proximal_vs_distal_usage'] = df.apply(lambda row: tuple([(p + isoform_pseudo_count) / (p + c + d + 2. * isoform_pseudo_count) for p, c, d in zip(list(row['proximal_count']), list(row['competing_count']), list(row['distal_count']))]), axis=1)
    df['proximal_vs_distal_logodds'] = df['proximal_vs_distal_usage'].apply(lambda t: tuple([np.log(p / (1. - p)) for p in list(t)]))
    df['pooled_proximal_vs_distal_usage'] = (df['pooled_proximal_count'] + pooled_isoform_pseudo_count) / (df['pooled_proximal_count'] + df['pooled_competing_count'] + df['pooled_distal_count'] + 2. * pooled_isoform_pseudo_count)
    df['mean_proximal_vs_distal_usage'] = df['proximal_vs_distal_usage'].apply(lambda t: np.mean(list(t)))
    df['median_proximal_vs_distal_usage'] = df['proximal_vs_distal_usage'].apply(lambda t: np.median(list(t)))
    df['pooled_proximal_vs_distal_logodds'] = np.log(df['pooled_proximal_vs_distal_usage'] / (1. - df['pooled_proximal_vs_distal_usage']))
    df['mean_proximal_vs_distal_logodds'] = df['proximal_vs_distal_logodds'].apply(lambda t: np.mean(list(t)))
    df['median_proximal_vs_distal_logodds'] = df['proximal_vs_distal_logodds'].apply(lambda t: np.median(list(t)))
    
    
    print('Dropping intermediate columns...')
    
    if drop_nans == True :
        df['pooled_proximal_logodds_is_nan'] = np.isnan(df['pooled_proximal_logodds']) | np.isinf(df['pooled_proximal_logodds'])
        df['mean_proximal_logodds_is_nan'] = np.isnan(df['mean_proximal_logodds']) | np.isinf(df['mean_proximal_logodds'])
        df['mean_proximal_avgcut_nan'] = np.isnan(df['mean_proximal_avgcut']) | np.isinf(df['mean_proximal_avgcut'])
        
        #df = df.query("pooled_proximal_logodds_is_nan == False and mean_proximal_logodds_is_nan == False").copy()# and mean_proximal_avgcut_nan == False
        df = df.query("pooled_proximal_logodds_is_nan == False").copy()# and mean_proximal_avgcut_nan == False
        
        df = df.drop(columns=['pooled_proximal_logodds_is_nan', 'mean_proximal_logodds_is_nan', 'mean_proximal_avgcut_nan'])

    
    df = df.drop(columns=['barcode', 'total_count', 'proximal_count', 'distal_count', 'proximal_usage', 'proximal_logodds', 'proximal_vs_distal_usage', 'proximal_vs_distal_logodds', 'cuts', 'proximal_avgcut'])
    
    df = df.reset_index()
    
    return df

def manual_df_processing(seq_df, clinvar_snv_df) :
    #Re-annotate SNV mutations against Clinvar
    clinvar_snv_df['master_seq'] = clinvar_snv_df['var'].str.slice(0, 164)
    clinvar_snv_df = clinvar_snv_df.set_index('master_seq')
    clinvar_snv_df = clinvar_snv_df[['significance', 'clinvar_id', 'observed_usage', 'in_acmg']]

    clinvar_snv_df = clinvar_snv_df.rename({'observed_usage' : 'apadb_usage'})


    seq_df['significance'] = 'Missing'
    seq_df['clinvar_id'] = 'Missing'
    seq_df['in_acmg'] = 'No'
    seq_df.loc[(seq_df.experiment == 'acmg_apadb') | (seq_df.experiment == 'acmg_polyadb'), 'in_acmg'] = 'Yes'
    seq_df['apadb_logodds'] = np.nan

    seq_df = seq_df.join(clinvar_snv_df, on='master_seq', how='left', rsuffix='_clinvarcopy').copy()

    valid_index = seq_df['clinvar_id_clinvarcopy'].notna()

    seq_df.loc[valid_index, 'clinvar_id'] = seq_df.loc[valid_index, 'clinvar_id_clinvarcopy']
    seq_df.loc[valid_index, 'significance'] = seq_df.loc[valid_index, 'significance_clinvarcopy']
    seq_df.loc[valid_index, 'in_acmg'] = seq_df.loc[valid_index, 'in_acmg_clinvarcopy']
    seq_df.loc[valid_index, 'apadb_logodds'] = seq_df.loc[valid_index, 'observed_usage']

    seq_df = seq_df.drop(columns=['clinvar_id_clinvarcopy', 'significance_clinvarcopy', 'in_acmg_clinvarcopy', 'observed_usage']).copy()



    #Re-map snv variants to wt sequences

    def hamming_distance(seq1, seq2) :
        dist = 0
        for j in range(0, len(seq1)) :
            if seq1[j] != seq2[j] :
                dist += 1

        return dist


    wt_dict = {}
    for index, row in seq_df.iterrows() :

        if row['variant'] == 'wt' :
            wt_gene = row['gene']
            if 'MAN_' in wt_gene :
                wt_gene = wt_gene.replace('MAN_', '')

            #wt_gene = wt_gene[:wt_gene.index('.')]

            if wt_gene not in wt_dict :
                wt_dict[wt_gene] = []

            wt_dict[wt_gene].append(row['master_seq'])

    #Append special case wt mappings
    if 'HBB.2' in wt_dict and 'HBB.3' in wt_dict :
        wt_dict['HBB.2'].extend(wt_dict['HBB.3'])

    wt_seqs = []
    for index, row in seq_df.iterrows() :

        wt_seq = row['wt_seq']

        if wt_seq == 'Unmapped' and row['gene'] in wt_dict :
            if row['variant'] == 'snv' :
                for wt_seq_candidate in wt_dict[row['gene']] :
                    if hamming_distance(row['master_seq'], wt_seq_candidate) == 1 :
                        wt_seq = wt_seq_candidate
                        break
            elif row['variant'] == 'indel' and len(wt_dict[row['gene']]) == 1 :
                if hamming_distance(row['master_seq'][:20], wt_dict[row['gene']][0][:20]) == 0 :
                    wt_seq = wt_dict[row['gene']][0]

        wt_seqs.append(wt_seq)

    seq_df['wt_seq'] = wt_seqs


    #Map TGTA variants to wt sequence

    tgta_wts = list(seq_df.query("experiment == 'tgta' and subexperiment == 'n=0'")['master_seq'].values)

    tgta_wts.extend(list(seq_df.loc[(seq_df.master_seq.str.contains('AGAGGATCAATCCCATCAGTGG')) & (seq_df.subexperiment == 'n=1')]['master_seq'].values))

    wt_seqs = []
    tgta_fixed = []
    for index, row in seq_df.iterrows() :

        wt_seq = row['wt_seq']

        if wt_seq == 'Unmapped' and row['experiment'] == 'tgta' :
            min_dist = 30
            min_wt_seq = 'Unmapped'
            for wt_seq_candidate in tgta_wts :
                hamming_dist = hamming_distance(row['master_seq'], wt_seq_candidate)

                if hamming_dist < min_dist :
                    min_dist = hamming_dist
                    min_wt_seq = wt_seq_candidate

            wt_seq = min_wt_seq

        wt_seqs.append(wt_seq)

        if 'AGAGGATCAATCCCATCAGTGG' in row['master_seq'] :
            tgta_fixed.append(True)
        else :
            tgta_fixed.append(False)

    seq_df['wt_seq'] = wt_seqs
    seq_df['tgta_fixed'] = tgta_fixed

    #Map TGTA mut positions

    tgta_pos_1_list = []
    tgta_pos_2_list = []
    tgta_pos_3_list = []


    for index, row in seq_df.iterrows() :

        tgta_pos_1 = 0
        tgta_pos_2 = 0
        tgta_pos_3 = 0

        if row['experiment'] == 'tgta' :

            tgta_start_pos = 0

            if row['subexperiment'] in ['n=1', 'n=2', 'n=3'] :

                for j in range(tgta_start_pos, len(row['master_seq']) - 3) :
                    if row['master_seq'][j:j+4] != row['wt_seq'][j:j+4] and row['master_seq'][j:j+4] == 'TGTA' :
                        tgta_start_pos = j
                        break

                tgta_pos_1 = tgta_start_pos
            if row['subexperiment'] in ['n=2', 'n=3'] :
                for j in range(tgta_start_pos + 4, len(row['master_seq']) - 3) :
                    if row['master_seq'][j:j+4] != row['wt_seq'][j:j+4] and row['master_seq'][j:j+4] == 'TGTA' :
                        tgta_start_pos = j
                        break

                tgta_pos_2 = tgta_start_pos

        tgta_pos_1_list.append(tgta_pos_1)
        tgta_pos_2_list.append(tgta_pos_2)
        tgta_pos_3_list.append(tgta_pos_3)

    seq_df['tgta_pos_1'] = tgta_pos_1_list
    seq_df['tgta_pos_2'] = tgta_pos_2_list
    seq_df['tgta_pos_3'] = tgta_pos_3_list
    
    return seq_df

#Manually annotate SNVs from HGMD
def manually_annotate_hgmd_variants(seq_df_delta) :
    seq_df_delta = seq_df_delta.set_index('master_seq')

    #F2.1
    seq_df_delta.loc['AACCAATCCCGTGAAAGAATTATTTTTGTGTTTCTAAAACTATGGTTCCCAATAAAAGTGACTCTCAGTGAGCCTCAATGCTCCCAGTGCTATTCATGGGCAGCTCTCTGGGCTCAGGAAGAGCCAGTAATACTACTGGATAAAGAAGACTTAAGAATCCACCA', 'significance'] = 'Undetermined'
    seq_df_delta.loc['AACCAATCCCGTGAAAGAATTATTTTTGTGTTTCTAAAACTATGGTTCCCAATAAAAGTGACTCTCAGTGAGCCTCAATGCTCCCAGTGCTATTCATGGGCAGCTCTCTGGGCTCAGGAAGAGCCAGTAATACTACTGGATAAAGAAGACTTAAGAATCCACCA', 'clinvar_id'] = 'c.*96C>T'
    seq_df_delta.loc['AACCAATCCCGTGAAAGAATTATTTTTGTGTTTCTAAAACTATGGTTCCCAATAAAAGTGACTCTCAGCGAGCCTCAAAGCTCCCAGTGCTATTCATGGGCAGCTCTCTGGGCTCAGGAAGAGCCAGTAATACTACTGGATAAAGAAGACTTAAGAATCCACCA', 'significance'] = 'Undetermined'
    seq_df_delta.loc['AACCAATCCCGTGAAAGAATTATTTTTGTGTTTCTAAAACTATGGTTCCCAATAAAAGTGACTCTCAGCGAGCCTCAAAGCTCCCAGTGCTATTCATGGGCAGCTCTCTGGGCTCAGGAAGAGCCAGTAATACTACTGGATAAAGAAGACTTAAGAATCCACCA', 'clinvar_id'] = 'c.*106T>A'
    seq_df_delta.loc['AACCAATCCCGTGAAAGAATTATTTTTGTGTTTCTAAAACTATGGTTCCCAATAAAAGTGACTCTCAGCGAGCCTCAATGTTCCCAGTGCTATTCATGGGCAGCTCTCTGGGCTCAGGAAGAGCCAGTAATACTACTGGATAAAGAAGACTTAAGAATCCACCA', 'significance'] = 'Pathogenic'
    seq_df_delta.loc['AACCAATCCCGTGAAAGAATTATTTTTGTGTTTCTAAAACTATGGTTCCCAATAAAAGTGACTCTCAGCGAGCCTCAATGTTCCCAGTGCTATTCATGGGCAGCTCTCTGGGCTCAGGAAGAGCCAGTAATACTACTGGATAAAGAAGACTTAAGAATCCACCA', 'clinvar_id'] = 'c.*108C>T'

    #HBA2.2
    seq_df_delta.loc['CTCCCAACGGGCCCTCCTCCCCTCCTTGCACCGGCCCTTCCTGGTCTTTGAATAAAGTCTGAGTGTGCAGCAGCCTGTGTGTGCCTGGGTTCTCTCTATCCCGGAATGTGCCAACAATGGAGGTGTTTACCTGTCTCAGACCAAGGACCTCTCTGCAGCTGCAT', 'significance'] = 'Pathogenic'
    seq_df_delta.loc['CTCCCAACGGGCCCTCCTCCCCTCCTTGCACCGGCCCTTCCTGGTCTTTGAATAAAGTCTGAGTGTGCAGCAGCCTGTGTGTGCCTGGGTTCTCTCTATCCCGGAATGTGCCAACAATGGAGGTGTTTACCTGTCTCAGACCAAGGACCTCTCTGCAGCTGCAT', 'clinvar_id'] = 'c.*104G>T'

    #seq_df_delta.loc['CTCCCAAAGGGCCCTCCTCCCCTCCTTGCACCGGCCCTTCCTGGTCTTTGAATAAAGTCTGAGTGGGCAGCAGCCTGTGTGTGCCTGGGTTCTCTCTATCCCGGAATGTGCCAACAATGGAGGTGTTTACCTGTCTCAGACCAAGGACCTCTCTGCAGCTGCAT', 'significance'] = 'Undetermined'
    #seq_df_delta.loc['CTCCCAAAGGGCCCTCCTCCCCTCCTTGCACCGGCCCTTCCTGGTCTTTGAATAAAGTCTGAGTGGGCAGCAGCCTGTGTGTGCCTGGGTTCTCTCTATCCCGGAATGTGCCAACAATGGAGGTGTTTACCTGTCTCAGACCAAGGACCTCTCTGCAGCTGCAT', 'clinvar_id'] = 'c.*46C>A'
    #seq_df_delta.loc['CTCCCAATGGGCCCTCCTCCCCTCCTTGCACCGGCCCTTCCTGGTCTTTGAATAAAGTCTGAGTGGGCAGCAGCCTGTGTGTGCCTGGGTTCTCTCTATCCCGGAATGTGCCAACAATGGAGGTGTTTACCTGTCTCAGACCAAGGACCTCTCTGCAGCTGCAT', 'significance'] = 'Undetermined'
    #seq_df_delta.loc['CTCCCAATGGGCCCTCCTCCCCTCCTTGCACCGGCCCTTCCTGGTCTTTGAATAAAGTCTGAGTGGGCAGCAGCCTGTGTGTGCCTGGGTTCTCTCTATCCCGGAATGTGCCAACAATGGAGGTGTTTACCTGTCTCAGACCAAGGACCTCTCTGCAGCTGCAT', 'clinvar_id'] = 'c.*46C>T'

    seq_df_delta.loc['CTCCCAACGGGCCCTCCTCCCCTCCTTGCACCGGCCCTTCCTGATCTTTGAATAAAGTCTGAGTGGGCAGCAGCCTGTGTGTGCCTGGGTTCTCTCTATCCCGGAATGTGCCAACAATGGAGGTGTTTACCTGTCTCAGACCAAGGACCTCTCTGCAGCTGCAT', 'significance'] = 'Undetermined'
    seq_df_delta.loc['CTCCCAACGGGCCCTCCTCCCCTCCTTGCACCGGCCCTTCCTGATCTTTGAATAAAGTCTGAGTGGGCAGCAGCCTGTGTGTGCCTGGGTTCTCTCTATCCCGGAATGTGCCAACAATGGAGGTGTTTACCTGTCTCAGACCAAGGACCTCTCTGCAGCTGCAT', 'clinvar_id'] = 'c.*82G>A'

    seq_df_delta.loc['CTCCCAACGGGCCCTCCTCCCCTCCTTGCACCGGCCCTTCCTGGTCTTTGAATAAAGTCCGAGTGGGCAGCAGCCTGTGTGTGCCTGGGTTCTCTCTATCCCGGAATGTGCCAACAATGGAGGTGTTTACCTGTCTCAGACCAAGGACCTCTCTGCAGCTGCAT', 'significance'] = 'Conflicting'
    seq_df_delta.loc['CTCCCAACGGGCCCTCCTCCCCTCCTTGCACCGGCCCTTCCTGGTCTTTGAATAAAGTCCGAGTGGGCAGCAGCCTGTGTGTGCCTGGGTTCTCTCTATCCCGGAATGTGCCAACAATGGAGGTGTTTACCTGTCTCAGACCAAGGACCTCTCTGCAGCTGCAT', 'clinvar_id'] = 'c.*98T>C'

    seq_df_delta.loc['CTCCCAACGGGCCCTCCTCCCCTCCTTGCACCGGCCCTTCCTGGTCTTTGAATAAAGTCTGAGTAGGCAGCAGCCTGTGTGTGCCTGGGTTCTCTCTATCCCGGAATGTGCCAACAATGGAGGTGTTTACCTGTCTCAGACCAAGGACCTCTCTGCAGCTGCAT', 'significance'] = 'Conflicting'
    seq_df_delta.loc['CTCCCAACGGGCCCTCCTCCCCTCCTTGCACCGGCCCTTCCTGGTCTTTGAATAAAGTCTGAGTAGGCAGCAGCCTGTGTGTGCCTGGGTTCTCTCTATCCCGGAATGTGCCAACAATGGAGGTGTTTACCTGTCTCAGACCAAGGACCTCTCTGCAGCTGCAT', 'clinvar_id'] = 'c.*103G>A'

    #PTEN.15
    seq_df_delta.loc['ATGTATATACCTTTTTGTGTCAAAAGGACATTTAAAATTCAATTAGGATTAATAAAGATGGCACTTTCCCATTTTATTCCAGTTTTATAAAAAGTGGAGACAGACTGATGTGTATACGTAGGAATTTTTTCCTTTTGTGTTCTGTCACCAACTGAAGTGGCTAA', 'significance'] = 'Likely benign'
    seq_df_delta.loc['ATGTATATACCTTTTTGTGTCAAAAGGACATTTAAAATTCAATTAGGATTAATAAAGATGGCACTTTCCCATTTTATTCCAGTTTTATAAAAAGTGGAGACAGACTGATGTGTATACGTAGGAATTTTTTCCTTTTGTGTTCTGTCACCAACTGAAGTGGCTAA', 'clinvar_id'] = 'c.*282G>A'

    #PTEN.16
    seq_df_delta.loc['TCTGAATTTTTTTTTATCAAGAGGGATAAAACACCATGAAAATAAACTTGAATAAACTGAAAATGGACCTTTTTTTTTCTAATGGCAATAGGACATTGTGTCAGATTACCAGTTATAGGAACAATTCTCTTTTCCTGACCAATCTTGTTTTACCCTATACATCC', 'significance'] = 'Undetermined'
    seq_df_delta.loc['TCTGAATTTTTTTTTATCAAGAGGGATAAAACACCATGAAAATAAACTTGAATAAACTGAAAATGGACCTTTTTTTTTCTAATGGCAATAGGACATTGTGTCAGATTACCAGTTATAGGAACAATTCTCTTTTCCTGACCAATCTTGTTTTACCCTATACATCC', 'clinvar_id'] = 'c.*74T>C'
    seq_df_delta.loc['TCTGAATTTTTTTTTATCAAGAGGGATAAAACACCATGAAAATAAACTTGAATAAACTGAAAATGGACCTTTTTTTTTTTAAGGGCAATAGGACATTGTGTCAGATTACCAGTTATAGGAACAATTCTCTTTTCCTGACCAATCTTGTTTTACCCTATACATCC', 'significance'] = 'Likely benign'
    seq_df_delta.loc['TCTGAATTTTTTTTTATCAAGAGGGATAAAACACCATGAAAATAAACTTGAATAAACTGAAAATGGACCTTTTTTTTTTTAAGGGCAATAGGACATTGTGTCAGATTACCAGTTATAGGAACAATTCTCTTTTCCTGACCAATCTTGTTTTACCCTATACATCC', 'clinvar_id'] = 'c.*78T>G'
    seq_df_delta.loc['TCTGAATTTTTTTTTATCAAGAGGGATAAAACACCATGAAAATAAACTTGAATAAACTGAAAATGGACCCTTTTTTTTTTAATGGCAATAGGACATTGTGTCAGATTACCAGTTATAGGAACAATTCTCTTTTCCTGACCAATCTTGTTTTACCCTATACATCC', 'significance'] = 'Undetermined'
    seq_df_delta.loc['TCTGAATTTTTTTTTATCAAGAGGGATAAAACACCATGAAAATAAACTTGAATAAACTGAAAATGGACCCTTTTTTTTTTAATGGCAATAGGACATTGTGTCAGATTACCAGTTATAGGAACAATTCTCTTTTCCTGACCAATCTTGTTTTACCCTATACATCC', 'clinvar_id'] = 'c.*65T>C'

    #BRCA1.1
    seq_df_delta.loc['ACTTGATTGTACAAAATACGTTTTGTAAATGTTGTGCTGTTAACACTGCAAATAATCTTGGTAGCAAACACTTCCACCATGAATGACTGTTCTTGAGACTTAGGCCAGCCGACTTTCTCAGAGCCTTTTCACTGTGCTTCAGTCTCCCACTCTGTAAAATGGGG', 'significance'] = 'Undetermined'
    seq_df_delta.loc['ACTTGATTGTACAAAATACGTTTTGTAAATGTTGTGCTGTTAACACTGCAAATAATCTTGGTAGCAAACACTTCCACCATGAATGACTGTTCTTGAGACTTAGGCCAGCCGACTTTCTCAGAGCCTTTTCACTGTGCTTCAGTCTCCCACTCTGTAAAATGGGG', 'clinvar_id'] = 'c.*1363A>T'

    #RNU4ATAC.4
    seq_df_delta.loc[seq_df_delta.gene == 'RNU4ATAC.4', 'significance'] = 'Pathogenic other'

    #HBB.1
    seq_df_delta.loc['TACTAAACTGGGGGATATTATGAAGGGCCTTGAGCATCTGGATTCTGCCTAATAAAAAACGTTTATTTTCATTGCAATGATGTATTTAAATTATTTCTGAATATTTTACTAAAAAGGGAATGTGGGAGGTCAGTGCATTTAAAACATAAAGAAATGAAGAGCTA', 'significance'] = 'Likely benign'
    seq_df_delta.loc['TACTAAACTGGGGGATATTATGAAGGGCCTTGAGCATCTGGATTCTGCCTAATAAAAAACGTTTATTTTCATTGCAATGATGTATTTAAATTATTTCTGAATATTTTACTAAAAAGGGAATGTGGGAGGTCAGTGCATTTAAAACATAAAGAAATGAAGAGCTA', 'clinvar_id'] = 'c.*118A>G'
    seq_df_delta.loc['TACTAAACTGGGGGATATTATGAAGGGCCTTGAGCATCTGGATTCTGCCTAATAAAAAACATTTATTTTCACTGCAATGATGTATTTAAATTATTTCTGAATATTTTACTAAAAAGGGAATGTGGGAGGTCAGTGCATTTAAAACATAAAGAAATGAAGAGCTA', 'significance'] = 'Likely benign'
    seq_df_delta.loc['TACTAAACTGGGGGATATTATGAAGGGCCTTGAGCATCTGGATTCTGCCTAATAAAAAACATTTATTTTCACTGCAATGATGTATTTAAATTATTTCTGAATATTTTACTAAAAAGGGAATGTGGGAGGTCAGTGCATTTAAAACATAAAGAAATGAAGAGCTA', 'clinvar_id'] = 'c.*129T>C'
    seq_df_delta.loc['TACTAAACTGGGGGATATTATGAAGGGCCTTGAGCATCTGGATTCTGCCTAATAAAAAACATTTATTTTCATTGAAATGATGTATTTAAATTATTTCTGAATATTTTACTAAAAAGGGAATGTGGGAGGTCAGTGCATTTAAAACATAAAGAAATGAAGAGCTA', 'significance'] = 'Likely benign'
    seq_df_delta.loc['TACTAAACTGGGGGATATTATGAAGGGCCTTGAGCATCTGGATTCTGCCTAATAAAAAACATTTATTTTCATTGAAATGATGTATTTAAATTATTTCTGAATATTTTACTAAAAAGGGAATGTGGGAGGTCAGTGCATTTAAAACATAAAGAAATGAAGAGCTA', 'clinvar_id'] = 'c.*132C>A'
    seq_df_delta.loc['TACTAAACTGGGGGATATTATGAAGGGCCTTGAGCATCTGGATTCTGCCTAATAAAAAACATTTATTTTCATTGTAATGATGTATTTAAATTATTTCTGAATATTTTACTAAAAAGGGAATGTGGGAGGTCAGTGCATTTAAAACATAAAGAAATGAAGAGCTA', 'significance'] = 'Undetermined'
    seq_df_delta.loc['TACTAAACTGGGGGATATTATGAAGGGCCTTGAGCATCTGGATTCTGCCTAATAAAAAACATTTATTTTCATTGTAATGATGTATTTAAATTATTTCTGAATATTTTACTAAAAAGGGAATGTGGGAGGTCAGTGCATTTAAAACATAAAGAAATGAAGAGCTA', 'clinvar_id'] = 'c.*132C>T'

    #ARSA.3
    seq_df_delta.loc['GCCTGTGGGGGAGGCTCAGGTGTCTGGAGGGGGTTTGTGCCTGATAACGTAATAACACTAGTGGAGACTTGCAGATGTGACAATTCGTCCAATCCTGGGGTAATGCTGTGTGCTGGTGCCGGTCCCCTGTGGTACGAATGAGGAAACTGAGGTGCAGAGAGGTT', 'significance'] = 'Undetermined'
    seq_df_delta.loc['GCCTGTGGGGGAGGCTCAGGTGTCTGGAGGGGGTTTGTGCCTGATAACGTAATAACACTAGTGGAGACTTGCAGATGTGACAATTCGTCCAATCCTGGGGTAATGCTGTGTGCTGGTGCCGGTCCCCTGTGGTACGAATGAGGAAACTGAGGTGCAGAGAGGTT', 'clinvar_id'] = 'c.*103C>T'
    seq_df_delta.loc['GCCTGTGGGGGAGGCTCAGGTGTCTGGAGGGGGTTTGTGCCTGATAACGTAATAACACCAGTGGAGACTTGCAGATGTGAGAATTCGTCCAATCCTGGGGTAATGCTGTGTGCTGGTGCCGGTCCCCTGTGGTACGAATGAGGAAACTGAGGTGCAGAGAGGTT', 'significance'] = 'Undetermined'
    seq_df_delta.loc['GCCTGTGGGGGAGGCTCAGGTGTCTGGAGGGGGTTTGTGCCTGATAACGTAATAACACCAGTGGAGACTTGCAGATGTGAGAATTCGTCCAATCCTGGGGTAATGCTGTGTGCTGGTGCCGGTCCCCTGTGGTACGAATGAGGAAACTGAGGTGCAGAGAGGTT', 'clinvar_id'] = 'c.*125C>G'
    seq_df_delta.loc['GCCTGTGGGGGAGGCTCAGGTGTCTGGAGGGGGTTTGTGCCTGATAACGTAATAACACCAGTGGAGACTTGCAGATGTGACAATTAGTCCAATCCTGGGGTAATGCTGTGTGCTGGTGCCGGTCCCCTGTGGTACGAATGAGGAAACTGAGGTGCAGAGAGGTT', 'significance'] = 'Undetermined'
    seq_df_delta.loc['GCCTGTGGGGGAGGCTCAGGTGTCTGGAGGGGGTTTGTGCCTGATAACGTAATAACACCAGTGGAGACTTGCAGATGTGACAATTAGTCCAATCCTGGGGTAATGCTGTGTGCTGGTGCCGGTCCCCTGTGGTACGAATGAGGAAACTGAGGTGCAGAGAGGTT', 'clinvar_id'] = 'c.*130C>A'

    return seq_df_delta.reset_index().copy()