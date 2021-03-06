from __future__ import print_function
import keras
from keras.models import Sequential, Model, load_model
from keras import backend as K

import tensorflow as tf

import pandas as pd

import os
import sys
import time
import pickle
import numpy as np

import scipy.sparse as sp
import scipy.io as spio

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import isolearn_keras as iso
from aparent_losses import *
from aparent_visualization import *

import websockets
import asyncio
import signal

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import json


#Load APADB pair data
'''apadb_pair_dict = pickle.load(open('apa_apadb_data.pickle', 'rb'))
apadb_pair_df = apadb_pair_dict['apadb_df']

#Take only pooled datapoints
apadb_pair_df = apadb_pair_df.query("tissue == 'pooled'").copy()

apadb_pair_df['seq_prox'] = apadb_pair_df['wide_seq_ext_prox'].str.slice(175-70, 175-70+205)
apadb_pair_df['seq_dist'] = apadb_pair_df['wide_seq_ext_dist'].str.slice(175-70, 175-70+205)
apadb_pair_df['rel_start_prox'] = apadb_pair_df['rel_start_prox'] #- 105
apadb_pair_df['rel_end_prox'] = apadb_pair_df['rel_end_prox'] + 1#- 105
apadb_pair_df['rel_start_dist'] = apadb_pair_df['rel_start_dist'] #- 105
apadb_pair_df['rel_end_dist'] = apadb_pair_df['rel_end_dist'] + 1#- 105
apadb_pair_df['site_distance'] = np.abs(apadb_pair_df['cut_start_prox'] - apadb_pair_df['cut_start_dist'])

gene_list = sorted(list(apadb_pair_df["gene"].unique()))
gene_id_list = sorted(list(apadb_pair_df["gene_id"].unique()))'''

#Load APADB data
apadb_df = pd.read_csv('leslie_apadb_data_wider_v2.csv', sep=',')

apadb_df['seq'] = apadb_df['wide_seq_ext'].str.slice(175-70, 175-70+205)

def get_start_pos(row) :
    if row['strand'] == '+' :
        return row['cut_start'] - row['pas_pos'] + 70
    else :
        return row['pas_pos'] - row['cut_end'] + 76

def get_end_pos(row) :
    if row['strand'] == '+' :
        return row['cut_end'] - row['pas_pos'] + 70 + 1
    else :
        return row['pas_pos'] - row['cut_start'] + 76 + 1

apadb_df['rel_start'] = apadb_df.apply(get_start_pos, axis=1)
apadb_df['rel_end'] = apadb_df.apply(get_end_pos, axis=1)

gene_list = sorted(list(apadb_df["gene"].unique()))
gene_id_list = sorted(list(apadb_df["gene_id"].unique()))

#Construct pair-wise APADB data
apadb_df['gene_id_dist'] = apadb_df['gene_id'].apply(lambda x: '.'.join(x.split('.')[:-1]) + '.' + str(int(x.split('.')[-1]) - 1))

df_dist = apadb_df.copy().set_index('gene_id')

dist_columns = [
    'sitenum',
    'pas',
    'seq',
    'wide_seq',
    'wide_seq_ext',
    'site_type',
    'pas_pos',
    'cut_start',
    'cut_end',
    'cut_mode',
    'mirna',
    'count',
    'rel_start',
    'rel_end'
]

df_dist = df_dist[dist_columns]

apadb_pair_df = apadb_df.join(df_dist, on='gene_id_dist', how='inner', lsuffix='_prox', rsuffix='_dist')
apadb_pair_df['site_distance'] = np.abs(apadb_pair_df['cut_start_prox'] - apadb_pair_df['cut_start_dist'])

pair_gene_list = sorted(list(apadb_pair_df["gene"].unique()))

#Load base APARENT model

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'aparent_plasmid_iso_cut_distalpas_large_lessdropout_all_libs_no_sampleweights.h5'
model_path = os.path.join(save_dir, model_name)

aparent_model = load_model(model_path)

#Load APADB-tuned APARENT model

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'aparent_apadb_fitted.h5'
model_path = os.path.join(save_dir, model_name)

apadb_model = load_model(model_path)

#Dummy compile
#opt = keras.optimizers.SGD(lr=0.1)
#model.compile(loss='mean_squared_error', optimizer=opt)

encoder = iso.OneHotEncoder(205)

def aparent_mutmap(seq, iso_start, iso_end) :
    cut_ref, _, _ = aparent_predict(seq, iso_start, iso_end)

    cut_vars = np.zeros((len(seq), 4, len(seq) + 1))

    for mut_pos in range(len(seq)) :
        for mut_nt_i, mut_nt in enumerate(['A', 'C', 'G', 'T']) :
            var_seq = seq[:mut_pos] + mut_nt + seq[mut_pos+1:]

            cut_pred, _, _ = aparent_predict(var_seq, iso_start, iso_end)

            cut_vars[mut_pos, mut_nt_i, :] = cut_pred[:]

    return cut_ref, cut_vars



def aparent_predict(seq, iso_start, iso_end) :
    #Predict iso and cut
    one_hot = np.reshape(encoder(seq), (1, 205, 4, 1))

    _, cut_pred = aparent_model.predict(x=[one_hot, np.zeros((1, 13)), np.ones((1, 1))])

    cut_pred = np.ravel(cut_pred)
    iso_pred = np.sum(cut_pred[iso_start: iso_end])
    logodds_pred = np.log(iso_pred / (1.0 - iso_pred))

    return cut_pred, iso_pred, logodds_pred

def apadb_predict(seq_prox, prox_cut_start, prox_cut_end, seq_dist, dist_cut_start, dist_cut_end, site_distance) :
    
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


async def hello(websocket, path):
    message = ''
    while message != 'exit' :
        message = await websocket.recv()
        print(f"< {message}")

        return_json = ''
        if 'aparent_' in message :
            _, seq, cut_start, cut_end = message.split("_")
            cut_start, cut_end = int(cut_start), int(cut_end)
        
            cut_pred, _, _ = aparent_predict(seq, cut_start, cut_end)

            return_json = json.dumps(
                {
                    "return_action" : "aparent",
                    #"cut_pred": str(["{:.6f}".format(cut) for cut in cut_pred.tolist()])
                    "cut_pred": [round(cut, 6) for cut in cut_pred.tolist()]
                }
            )
        elif 'variant_' in message :
            _, ref_seq, var_seq, cut_start, cut_end = message.split("_")
            cut_start, cut_end = int(cut_start), int(cut_end)
        
            cut_ref, _, _ = aparent_predict(ref_seq, cut_start, cut_end)
            cut_var, _, _ = aparent_predict(var_seq, cut_start, cut_end)

            return_json = json.dumps(
                {
                    "return_action" : "variant",
                    "cut_ref": [round(cut, 6) for cut in cut_ref.tolist()],
                    "cut_var": [round(cut, 6) for cut in cut_var.tolist()]
                }
            )
        elif 'mutmap_' in message :
            _, ref_seq, cut_start, cut_end = message.split("_")
            cut_start, cut_end = int(cut_start), int(cut_end)
        
            cut_ref, cut_vars = aparent_mutmap(ref_seq, cut_start, cut_end)

            return_json = json.dumps(
                {
                    "return_action" : "mutmap",
                    "cut_ref": [round(cut, 6) for cut in cut_ref.tolist()],
                    "cut_vars": np.round(cut_vars, 6).tolist()
                }
            )
        elif 'apadb_' in message :
            _, seq_prox, prox_cut_start, prox_cut_end, seq_dist, dist_cut_start, dist_cut_end, site_distance = message.split("_")
            prox_cut_start, prox_cut_end, dist_cut_start, dist_cut_end, site_distance = int(prox_cut_start), int(prox_cut_end), int(dist_cut_start), int(dist_cut_end), int(site_distance)
        
            iso_pred, cut_prox, cut_dist = apadb_predict(seq_prox, prox_cut_start, prox_cut_end, seq_dist, dist_cut_start, dist_cut_end, site_distance)

            return_json = json.dumps(
                {
                    "return_action" : "apadb",
                    "iso" : str(round(iso_pred, 6)),
                    "cut_prox" : [round(cut, 6) for cut in cut_prox.tolist()],
                    "cut_dist" : [round(cut, 6) for cut in cut_dist.tolist()]
                }
            )
        elif 'getsites_' in message :
            _, gene = message.split("_")

            gene_df = apadb_pair_df.query("gene == '" + gene + "'")

            return_json = json.dumps(
                {
                    "return_action" : "getsites",
                    "gene" : [str(row["gene"]) for _, row in gene_df.iterrows()],
                    "gene_id" : [str(row["gene_id"]) for _, row in gene_df.iterrows()],
                    "sitenum_prox" : [str(row["sitenum_prox"]) for _, row in gene_df.iterrows()],
                    "sitenum_dist" : [str(row["sitenum_dist"]) for _, row in gene_df.iterrows()],
                    "site_type_prox" : [str(row["site_type_prox"]) for _, row in gene_df.iterrows()],
                    "site_type_dist" : [str(row["site_type_dist"]) for _, row in gene_df.iterrows()]
                }
            )
        elif 'getseqs_' in message :
            _, gene_id = message.split("_")

            gene_df = apadb_pair_df.query("gene_id == '" + gene_id + "'")

            return_json = json.dumps(
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
                    "cut_start_prox" : str(gene_df["rel_start_prox"].values[0]),
                    "cut_end_prox" : str(gene_df["rel_end_prox"].values[0]),
                    "cut_start_dist" : str(gene_df["rel_start_dist"].values[0]),
                    "cut_end_dist" : str(gene_df["rel_end_dist"].values[0]),
                    "cut_start_coord_prox" : str(gene_df["cut_start_prox"].values[0]),
                    "cut_end_coord_prox" : str(gene_df["cut_end_prox"].values[0]),
                    "cut_start_coord_dist" : str(gene_df["cut_start_dist"].values[0]),
                    "cut_end_coord_dist" : str(gene_df["cut_end_dist"].values[0])
                }
            )
        elif 'getevents_' in message :
            _, gene = message.split("_")

            gene_df = apadb_df.query("gene == '" + gene + "'")

            return_json = json.dumps(
                {
                    "return_action" : "getevents",
                    "gene" : [str(row["gene"]) for _, row in gene_df.iterrows()],
                    "gene_id" : [str(row["gene_id"]) for _, row in gene_df.iterrows()],
                    "sitenum" : [str(row["sitenum"]) for _, row in gene_df.iterrows()],
                    "site_type" : [str(row["site_type"]) for _, row in gene_df.iterrows()]
                }
            )
        elif 'getseq_' in message :
            _, gene_id = message.split("_")

            gene_df = apadb_df.query("gene_id == '" + gene_id + "'")

            return_json = json.dumps(
                {
                    "return_action" : "getseq",
                    "gene" : str(gene_df["gene"].values[0]),
                    "gene_id" : str(gene_df["gene_id"].values[0]),
                    "chrom" : str(gene_df["chrom"].values[0]),
                    "strand" : str(gene_df["strand"].values[0]),
                    "sitenum" : str(gene_df["sitenum"].values[0]),
                    "site_type" : str(gene_df["site_type"].values[0]),
                    "seq" : str(gene_df["seq"].values[0]),
                    "cut_start" : str(gene_df["rel_start"].values[0]),
                    "cut_end" : str(gene_df["rel_end"].values[0]),
                    "chrom" : str(gene_df["chrom"].values[0]),
                    "strand" : str(gene_df["strand"].values[0]),
                    "cut_start_coord" : str(gene_df["cut_start"].values[0]),
                    "cut_end_coord" : str(gene_df["cut_end"].values[0])
                }
            )
        elif 'getgenes' == message :
            return_json = json.dumps(
                {
                    "return_action" : "getgenes",
                    "genes" : gene_list
                }
            )
        elif 'getpairgenes' == message :
            return_json = json.dumps(
                {
                    "return_action" : "getgenes",
                    "genes" : pair_gene_list
                }
            )


        await websocket.send(return_json)
        print(f"> {return_json}")

loop = asyncio.get_event_loop()


# Create the server.
start_server = websockets.serve(hello, 'localhost', 9990)
server = loop.run_until_complete(start_server)

# Run the server until receiving SIGTERM.
stop = asyncio.Future()
loop.add_signal_handler(signal.SIGTERM, stop.set_result, None)
loop.run_until_complete(stop)

# Shut down the server.
server.close()
loop.run_until_complete(server.wait_closed())
