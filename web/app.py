from flask import Flask, render_template, jsonify, request, send_from_directory

import os

import pandas as pd
import numpy as np

import keras
from keras.models import Sequential, Model, load_model
import tensorflow as tf

from load_apadb_model_helper import build_apadb_model


app = Flask(__name__, static_url_path='')

@app.route('/img/<path:filename>')
def serve_static_img(filename):
    #root_dir = os.path.dirname(os.getcwd())
    #return send_from_directory(os.path.join(root_dir, 'static', 'img'), filename)
    #return "test"
    return send_from_directory(os.path.join('static', 'img'), filename)

@app.route('/')
def home_view():
    return render_template('home_bootstrap_v2.html')

@app.route('/mutagenesis')
def mutagenesis_view():
    query_parameters = request.args
    use_zoom = query_parameters.get('zoom')

    if use_zoom is None or use_zoom == '' or use_zoom != 'false' :
        return render_template('saturation_mutagenesis_bootstrap_v2_zoom.html')
    else :
        return render_template('saturation_mutagenesis_bootstrap_v2.html')

@app.route('/apa')
def apa_view():
    query_parameters = request.args
    use_zoom = query_parameters.get('zoom')

    if use_zoom is None or use_zoom == '' or use_zoom != 'false' :
        return render_template('native_apa_bootstrap_v2_zoom.html')
    else :
        return render_template('native_apa_bootstrap_v2.html')

@app.route('/pas')
def pas_view():
    query_parameters = request.args
    use_zoom = query_parameters.get('zoom')

    if use_zoom is None or use_zoom == '' or use_zoom != 'false' :
        return render_template('pas_score_bootstrap_v2_zoom.html')
    else :
        return render_template('pas_score_bootstrap_v2.html')

@app.route('/variant')
def variant_view():
    query_parameters = request.args
    use_zoom = query_parameters.get('zoom')

    if use_zoom is None or use_zoom == '' or use_zoom != 'false' :
        return render_template('pas_variant_bootstrap_v2_zoom.html')
    else :
        return render_template('pas_variant_bootstrap_v2.html')

class OneHotEncoder :
    def __init__(self, seq_length=100, default_fill_value=0) :
        self.seq_length = seq_length
        self.default_fill_value = default_fill_value
        self.encode_map = {
            'A' : 0,
            'C' : 1,
            'G' : 2,
            'T' : 3
        }
        self.decode_map = {
            0 : 'A',
            1 : 'C',
            2 : 'G',
            3 : 'T',
            -1 : 'X'
        }
    
    def encode(self, seq) :
        one_hot = np.zeros((self.seq_length, 4))
        self.encode_inplace(seq, one_hot)

        return one_hot
    
    def encode_inplace(self, seq, encoding) :
        for pos, nt in enumerate(list(seq)) :
            if nt in self.encode_map :
                encoding[pos, self.encode_map[nt]] = 1
            elif self.default_fill_value != 0 :
                encoding[pos, :] = self.default_fill_value
    
    def __call__(self, seq) :
        return self.encode(seq)


file_path_prefix = "/www/research/apa/hello/"

#Load PolyaDB data
polyadb_df = pd.read_csv(file_path_prefix + 'data/web_polyadb_data.csv', sep='\t')
polyadb_pair_df = pd.read_csv(file_path_prefix + 'data/web_polyadb_pair_data.csv', sep='\t')

polyadb_gene_list = sorted(list(polyadb_df["gene"].unique()))
polyadb_gene_id_list = sorted(list(polyadb_df["gene_id"].unique()))
polyadb_pair_gene_list = sorted(list(polyadb_pair_df["gene"].unique()))

#Load ClinVar data
clinvar_polyadb_df = pd.read_csv(file_path_prefix + 'data/web_clinvar_polyadb_data.csv', sep='\t')

#Load APADB data
apadb_df = pd.read_csv(file_path_prefix + 'data/web_apadb_data.csv', sep='\t')
apadb_pair_df = pd.read_csv(file_path_prefix + 'data/web_apadb_pair_data.csv', sep='\t')

apadb_gene_list = sorted(list(apadb_df["gene"].unique()))
apadb_gene_id_list = sorted(list(apadb_df["gene_id"].unique()))
apadb_pair_gene_list = sorted(list(apadb_pair_df["gene"].unique()))

#Load ClinVar data
clinvar_apadb_df = pd.read_csv(file_path_prefix + 'data/web_clinvar_apadb_data.csv', sep='\t')


db_dict = {
    'polyadb' : {
        'df' : polyadb_df,
        'pair_df' : polyadb_pair_df,
        'gene_list' : polyadb_gene_list,
        'gene_id_list' : polyadb_gene_id_list,
        'pair_gene_list' : polyadb_pair_gene_list,
        'clinvar_df' : clinvar_polyadb_df
    },
    'apadb' : {
        'df' : apadb_df,
        'pair_df' : apadb_pair_df,
        'gene_list' : apadb_gene_list,
        'gene_id_list' : apadb_gene_id_list,
        'pair_gene_list' : apadb_pair_gene_list,
        'clinvar_df' : clinvar_apadb_df
    }
}




#Load base APARENT model
model_path = file_path_prefix + 'saved_models/aparent_large_lessdropout_all_libs_no_sampleweights.h5'
aparent_model = load_model(model_path)

#Load APADB-tuned APARENT model
apadb_model = build_apadb_model()
model_path = file_path_prefix + 'saved_models/aparent_apadb_fitted_large_lessdropout_no_sampleweights.h5'
apadb_model.load_weights(model_path)
#apadb_model = load_model(model_path)



encoder = OneHotEncoder(205)

aparent_model._make_predict_function()
apadb_model._make_predict_function()

def aparent_mutmap(encoder, aparent_model, seq, iso_start, iso_end) :
    #Clean inputs
    seq = (seq + ('X' * 205))[:205]
    if iso_start < 0 or iso_start >= 205 :
        iso_start = 0
    if iso_end < 0 or iso_end >= 205 :
        iso_end = 205 - 1
    
    cut_ref, _, _ = aparent_predict(encoder, aparent_model, seq, iso_start, iso_end)

    cut_vars = np.zeros((len(seq), 4, len(seq) + 1))

    for mut_pos in range(len(seq)) :
        for mut_nt_i, mut_nt in enumerate(['A', 'C', 'G', 'T']) :
            var_seq = seq[:mut_pos] + mut_nt + seq[mut_pos+1:]

            cut_pred, _, _ = aparent_predict(encoder, aparent_model, var_seq, iso_start, iso_end)

            cut_vars[mut_pos, mut_nt_i, :] = cut_pred[:]

    return cut_ref, cut_vars



def aparent_predict(encoder, aparent_model, seq, iso_start, iso_end) :
    #Clean inputs
    seq = (seq + ('X' * 205))[:205]
    if iso_start < 0 or iso_start >= 205 :
        iso_start = 0
    if iso_end < 0 or iso_end >= 205 :
        iso_end = 205 - 1

    #Predict iso and cut
    one_hot = np.reshape(encoder(seq), (1, 205, 4, 1))

    _, cut_pred = aparent_model.predict(x=[one_hot, np.zeros((1, 13)), np.ones((1, 1))])

    cut_pred = np.ravel(cut_pred)
    iso_pred = np.sum(cut_pred[iso_start: iso_end])
    logodds_pred = np.log(iso_pred / (1.0 - iso_pred))

    return cut_pred, iso_pred, logodds_pred

def apadb_predict(encoder, apadb_model, seq_prox, prox_cut_start, prox_cut_end, seq_dist, dist_cut_start, dist_cut_end, site_distance) :
    #Clean inputs
    seq_prox = (seq_prox + ('X' * 205))[:205]
    if prox_cut_start < 0 or prox_cut_start >= 205 :
        prox_cut_start = 0
    if prox_cut_end < 0 or prox_cut_end >= 205 :
        prox_cut_end = 205 - 1

    seq_dist = (seq_dist + ('X' * 205))[:205]
    if dist_cut_start < 0 or dist_cut_start >= 205 :
        dist_cut_start = 0
    if dist_cut_end < 0 or dist_cut_end >= 205 :
        dist_cut_end = 205 - 1

    if site_distance < 40 :
        site_distance = 40
    if site_distance > 1000000 :
        site_distance = 1000000

    site_distance = np.log(np.array([site_distance]).reshape(1, -1)) #Site distance in log-space

    prox_cut_start = np.array([prox_cut_start]).reshape(1, -1)
    prox_cut_end = np.array([prox_cut_end]).reshape(1, -1)
    dist_cut_start = np.array([dist_cut_start]).reshape(1, -1)
    dist_cut_end = np.array([dist_cut_end]).reshape(1, -1)

    onehot_prox = np.reshape(encoder(seq_prox), (1, len(seq_prox), 4, 1))
    onehot_dist = np.reshape(encoder(seq_dist), (1, len(seq_dist), 4, 1))

    #Predict with APADB-tuned APARENT model
    iso_pred, cut_prox, cut_dist = apadb_model.predict(x=[onehot_prox, onehot_dist, prox_cut_start, prox_cut_end, dist_cut_start, dist_cut_end, site_distance, np.zeros((1, 13)), np.ones((1, 1))])

    return iso_pred[0, 0], np.ravel(cut_prox), np.ravel(cut_dist)




@app.route('/api/genes', methods=['GET'])
def get_genes() :

    query_parameters = request.args
    database_id = query_parameters.get('database')

    db = db_dict['polyadb']
    if database_id is not None and database_id != '' :
        db = db_dict[database_id]

    return jsonify({
    	"genes" : db['gene_list']
    })

@app.route('/api/pairgenes', methods=['GET'])
def get_pairgenes() :

    query_parameters = request.args
    database_id = query_parameters.get('database')

    db = db_dict['polyadb']
    if database_id is not None and database_id != '' :
        db = db_dict[database_id]

    return jsonify({
        "genes" : db['pair_gene_list']
    })

@app.route('/api/genes/<string:gene>', methods=['GET'])
def get_sites(gene) :

    query_parameters = request.args
    database_id = query_parameters.get('database')

    db = db_dict['polyadb']
    if database_id is not None and database_id != '' :
        db = db_dict[database_id]

    gene_df = db['df'].query("gene == '" + gene + "'")

    return jsonify(
        {
            "gene" : [str(row["gene"]) for _, row in gene_df.iterrows()],
            "gene_id" : [str(row["gene_id"]) for _, row in gene_df.iterrows()],
            "sitenum" : [str(row["sitenum"]) for _, row in gene_df.iterrows()],
            "site_type" : [str(row["site_type"]) for _, row in gene_df.iterrows()]
        }
    )

@app.route('/api/sites/<string:gene_id>', methods=['GET'])
def get_site(gene_id) :

    query_parameters = request.args
    database_id = query_parameters.get('database')

    db = db_dict['polyadb']
    if database_id is not None and database_id != '' :
        db = db_dict[database_id]

    gene_df = db['df'].query("gene_id == '" + gene_id + "'")

    clinvar_gene_df = db['clinvar_df'].query("clinvar_gene == '" + gene_id.split(".")[0] + "'")

    return jsonify(
        {
            "gene" : str(gene_df["gene"].values[0]),
            "gene_id" : str(gene_df["gene_id"].values[0]),
            "chrom" : str(gene_df["chrom"].values[0]),
            "strand" : str(gene_df["strand"].values[0]),
            "sitenum" : str(gene_df["sitenum"].values[0]),
            "site_type" : str(gene_df["site_type"].values[0]),
            "seq" : str(gene_df["seq"].values[0]),
            "chrom" : str(gene_df["chrom"].values[0]),
            "strand" : str(gene_df["strand"].values[0]),
            "pas_pos_coord" : str(gene_df["pas_pos"].values[0]),
            "native_usage" : str(gene_df["ratio"].values[0]),
            "clinvar_dict" : {
                row['flat_id'] : {
                    'id' : row['clinvar_id'],
                    'significance' : row['clinvar_significance']
                }
                for _, row in clinvar_gene_df.iterrows()
            }
        }
    )

@app.route('/api/pairgenes/<string:gene>', methods=['GET'])
def get_pairsites(gene) :

    query_parameters = request.args
    database_id = query_parameters.get('database')

    db = db_dict['polyadb']
    if database_id is not None and database_id != '' :
        db = db_dict[database_id]

    gene_df = db['pair_df'].query("gene == '" + gene + "'")

    return jsonify(
        {
            "gene" : [str(row["gene"]) for _, row in gene_df.iterrows()],
            "gene_id" : [str(row["gene_id"]) for _, row in gene_df.iterrows()],
            "sitenum_prox" : [str(row["sitenum_prox"]) for _, row in gene_df.iterrows()],
            "sitenum_dist" : [str(row["sitenum_dist"]) for _, row in gene_df.iterrows()],
            "site_type_prox" : [str(row["site_type_prox"]) for _, row in gene_df.iterrows()],
            "site_type_dist" : [str(row["site_type_dist"]) for _, row in gene_df.iterrows()]
        }
    )

@app.route('/api/pairsites/<string:gene_id>', methods=['GET'])
def get_pairsite(gene_id) :

    query_parameters = request.args
    database_id = query_parameters.get('database')

    db = db_dict['polyadb']
    if database_id is not None and database_id != '' :
        db = db_dict[database_id]

    gene_df = db['pair_df'].query("gene_id == '" + gene_id + "'")

    return jsonify(
        {
            "return_action" : "getseqs",
            "gene" : str(gene_df["gene"].values[0]),
            "gene_id" : str(gene_df["gene_id"].values[0]),
            "chrom" : str(gene_df["chrom"].values[0]),
            "strand" : str(gene_df["strand"].values[0]),
            "sitenum_prox" : str(gene_df["sitenum_prox"].values[0]),
            "sitenum_dist" : str(gene_df["sitenum_dist"].values[0]),
            "site_type_prox" : str(gene_df["site_type_prox"].values[0]),
            "site_type_dist" : str(gene_df["site_type_dist"].values[0]),
            "seq_prox" : str(gene_df["seq_prox"].values[0]),
            "seq_dist" : str(gene_df["seq_dist"].values[0]),
            "site_distance" : str(gene_df["site_distance"].values[0]),
            "pas_pos_coord_prox" : str(gene_df["pas_pos_prox"].values[0]),
            "pas_pos_coord_dist" : str(gene_df["pas_pos_dist"].values[0]),
            "native_usage" : str(gene_df["ratio"].values[0])
        }
    )



@app.route('/api/aparent', methods=['GET'])
def predict_aparent() :

    query_parameters = request.args

    seq = query_parameters.get('sequence')
    cut_start = int(query_parameters.get('cut_start'))
    cut_end = int(query_parameters.get('cut_end'))

    cut_pred, _, _ = aparent_predict(encoder, aparent_model, seq, cut_start, cut_end)

    return jsonify(
        {
            "cut_pred": [round(cut, 6) for cut in cut_pred.tolist()]
        }
    )

@app.route('/api/variant', methods=['GET'])
def predict_variant() :

    query_parameters = request.args

    ref_seq = query_parameters.get('ref_sequence')
    var_seq = query_parameters.get('var_sequence')
    cut_start = int(query_parameters.get('cut_start'))
    cut_end = int(query_parameters.get('cut_end'))

    cut_ref, _, _ = aparent_predict(encoder, aparent_model, ref_seq, cut_start, cut_end)
    cut_var, _, _ = aparent_predict(encoder, aparent_model, var_seq, cut_start, cut_end)

    return jsonify(
        {
            "cut_ref": [round(cut, 6) for cut in cut_ref.tolist()],
            "cut_var": [round(cut, 6) for cut in cut_var.tolist()]
        }
    )

@app.route('/api/apadb', methods=['GET'])
def predict_apadb() :

    query_parameters = request.args

    seq_prox = query_parameters.get('seq_prox')
    prox_cut_start = int(query_parameters.get('prox_cut_start'))
    prox_cut_end = int(query_parameters.get('prox_cut_end'))
    seq_dist = query_parameters.get('seq_dist')
    dist_cut_start = int(query_parameters.get('dist_cut_start'))
    dist_cut_end = int(query_parameters.get('dist_cut_end'))
    site_distance = int(query_parameters.get('site_distance'))

    iso_pred, cut_prox, cut_dist = apadb_predict(encoder, apadb_model, seq_prox, prox_cut_start, prox_cut_end, seq_dist, dist_cut_start, dist_cut_end, site_distance)

    return jsonify(
        {
            "iso" : str(round(iso_pred, 6)),
            "cut_prox" : [round(cut, 6) for cut in cut_prox.tolist()],
            "cut_dist" : [round(cut, 6) for cut in cut_dist.tolist()]
        }
    )

@app.route('/api/mutagenesis', methods=['GET'])
def predict_mutagenesis() :

    query_parameters = request.args

    ref_seq = query_parameters.get('sequence')
    cut_start = 80
    cut_end = 115

    cut_ref, cut_vars = aparent_mutmap(encoder, aparent_model, ref_seq, cut_start, cut_end)

    return jsonify(
        {
            "cut_ref": [round(cut, 6) for cut in cut_ref.tolist()],
            "cut_vars": np.round(cut_vars, 6).tolist()
        }
    )

