import numpy as np

import matplotlib.collections as collections
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

import isolearn.keras as iso

#Helper functions

def logit(x) :
    return np.log(x / (1.0 - x))

def aparent_single_example_batch(one_hot) :
    return [
        np.reshape(one_hot, (1, one_hot.shape[0], one_hot.shape[1], 1)),
        np.zeros((1, 13)),
        np.ones((1, 1))
    ]

def predict_ref(model, seq, isoform_start=80, isoform_end=105) :
    one_hot = iso.OneHotEncoder(len(seq))(seq)
    _, cut_pred = model.predict(x=aparent_single_example_batch(one_hot))
        
    return np.sum(np.ravel(cut_pred)[isoform_start: isoform_end]), np.ravel(cut_pred)

def predict_mut_map(model, seq, isoform_start=80, isoform_end=105) :
    encoder = iso.OneHotEncoder(len(seq))
    
    mut_map = np.zeros((len(seq), 4))
    cut_map = np.zeros((len(seq), 4, len(seq) + 1))
    
    for pos in range(len(seq)) :
        for j, nt in enumerate(['A', 'C', 'G', 'T']) :
            mut_seq = seq[:pos] + nt + seq[pos+1:]
            one_hot = encoder(mut_seq)
    
            _, cut_pred = model.predict(x=aparent_single_example_batch(one_hot))
            
            mut_map[pos, j] = np.sum(np.ravel(cut_pred)[isoform_start: isoform_end])
            cut_map[pos, j, :] = np.ravel(cut_pred)
        
    return mut_map, cut_map

#Keras visualization and mutation maps

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

def mut_map(model, ref_seq, cut_snvs, seq_start, seq_end, isoform_start, isoform_end, figsize=(12, 6), height_ratios=[6, 2, 2], bg_alpha=0.5, border_eta = 0.085, logodds_clip=1.5, fig_name=None, fig_dpi=300) :

    ref_iso, ref_cut = predict_ref(model, ref_seq, isoform_start, isoform_end)
    
    mut_map, cut_map = predict_mut_map(model, ref_seq, isoform_start, isoform_end)
    
    mut_map = np.log(mut_map / (1.0 - mut_map)) - np.log(ref_iso / (1.0 - ref_iso))
    
    mut_map = mut_map.T
    mut_map = np.flipud(mut_map)
    
    #Slice according to seq trim index
    ref_seq = ref_seq[seq_start: seq_end]
    mut_map = mut_map[:, seq_start: seq_end]
    cut_map = cut_map[seq_start: seq_end, :, seq_start: seq_end]
    ref_cut = ref_cut[seq_start: seq_end]
    
    fig = plt.figure(figsize=figsize)
    
    gs = gridspec.GridSpec(3, 1, height_ratios=height_ratios)

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    ax = [ax0, ax1, ax2]
    
    mut_map = np.clip(mut_map, -logodds_clip, logodds_clip)
    
    bias = np.max(np.sum(mut_map[:, :], axis=0)) / 3.0 + 0.5
    max_score = np.min(np.sum(mut_map[:, :], axis=0)) / 3.0 * -1 + bias

    for i in range(0, seq_end - seq_start) :
        mutability_score = np.sum(mut_map[:, i]) / 3.0 * -1 + bias
        
        letterAt(ref_seq[i], i + 0.5, 0, mutability_score, ax[1])

    ax[1].plot([0, mut_map.shape[1]], [bias, bias], color='black', linestyle='--')

    plt.sca(ax[1])
    plt.xlim((0, seq_end - seq_start))
    plt.ylim((0, max_score))
    plt.axis('off')
    plt.yticks([0.5, bias, max_score], [round(bias - 0.5, 2), 0, round((max_score - bias) * -1, 2)], fontsize=16)

    pcm = ax[2].pcolor(mut_map, cmap='RdBu_r', vmin=-np.abs(mut_map).max(), vmax=np.abs(mut_map).max())
    plt.sca(ax[2])

    plt.xticks([], [])
    plt.yticks([0.5, 1.5, 2.5, 3.5], ['T', 'G', 'C', 'A'], fontsize=14)

    plt.axis([0, seq_end - seq_start, 0, 4])
    
    max_y_var_hat = 0
    for snv_pos, snv_nt, snv_color in cut_snvs :
        snv_pos -= seq_start
        
        base = 0
        if snv_nt == 'A' :
            base = 0
        elif snv_nt == 'C' :
            base = 1
        elif snv_nt == 'G' :
            base = 2
        elif snv_nt == 'T' :
            base = 3
        
        var_cut = cut_map[snv_pos, base, :]
        
        max_y_var_hat = max(max_y_var_hat, np.max(var_cut))
        
        ax[0].plot(np.arange(seq_end - seq_start), var_cut, color=snv_color, linestyle='--', linewidth=3, alpha=0.7)
        
        ax[2].add_patch(Rectangle((snv_pos + border_eta, 3 - base + border_eta), 1 - 2.*border_eta, 1 - 2.*border_eta, fill=False, edgecolor=snv_color, lw=4))
    
    max_y_var_hat = max(max_y_var_hat, np.max(ref_cut))
    
    ax[0].plot(np.arange(seq_end - seq_start), ref_cut, color='black', linestyle='-', linewidth=3, alpha=0.7)
    
    ax[0].plot([isoform_start - seq_start, isoform_start - seq_start], [0, max_y_var_hat], color='darkgreen', linestyle='--', linewidth=3, alpha=0.7)
    ax[0].plot([isoform_end - seq_start, isoform_end - seq_start], [0, max_y_var_hat], color='darkgreen', linestyle='--', linewidth=3, alpha=0.7)

    plt.sca(ax[0])
    plt.xlim((0, seq_end - seq_start))
    plt.xticks([70 - seq_start, 76- seq_start], ['CSE', 'CSE+6'],fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, max_y_var_hat * 1.02)

    #plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    #plt.tight_layout()
    
    if fig_name is not None :
        plt.savefig(fig_name + '.png', transparent=True, dpi=fig_dpi)
        plt.savefig(fig_name + '.svg')
        plt.savefig(fig_name + '.eps')
    plt.show()
