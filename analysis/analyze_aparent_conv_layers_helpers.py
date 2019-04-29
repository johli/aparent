import numpy as np

import scipy.sparse as sp
import scipy.io as spio

import isolearn.io as isoio

from scipy.stats import pearsonr

import matplotlib.pyplot as plt

import matplotlib.cm as cm
import matplotlib.colors as colors

import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter

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



def plot_pwm_iso_logo(pwms, r_vals, k, logo_ax, corr_ax, seq_start=4, seq_end=101, cse_start=49) :
    #Make sequence logo
    pwm = pwms[k]
    
    entropy = np.zeros(pwm.shape)
    entropy[pwm > 0] = pwm[pwm > 0] * -np.log2(pwm[pwm > 0])
    entropy = np.sum(entropy, axis=1)
    conservation = 2 - entropy

    height_base = 0.0
    logo_height = 1.0
    
    for j in range(pwm.shape[0]) :
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

            if ii == 0 :
                letterAt(nt, j + 0.5, height_base, nt_prob * logo_height, logo_ax, color=None)
            else :
                prev_prob = np.sum(pwm[j, sort_index[:ii]] * conservation[j]) * logo_height
                letterAt(nt, j + 0.5, height_base + prev_prob, nt_prob * logo_height, logo_ax, color=None)

    plt.sca(logo_ax)
    plt.xlim((0, pwm.shape[0]))
    plt.ylim((0, 2))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.axis('off')
    logo_ax.axhline(y=0.01 + height_base, color='black', linestyle='-', linewidth=2)
    
    #Make correlation map
    r_val_vec = r_vals[k, seq_start: seq_end].reshape(1, -1)
    corr_ax.imshow(r_val_vec, cmap='RdBu_r', aspect='auto', vmin=-np.max(np.abs(r_vals[:, seq_start: seq_end])), vmax=np.max(np.abs(r_vals[:, seq_start: seq_end])))

    plt.sca(corr_ax)
    plt.xticks([cse_start - seq_start], ['Start of CSE'], fontsize=12)
    plt.yticks([], [])


def plot_pwm_cut_logo(pwms, r_vals, k, logo_ax, corr_ax) :
    #Make sequence logo
    pwm = pwms[k]
    
    entropy = np.zeros(pwm.shape)
    entropy[pwm > 0] = pwm[pwm > 0] * -np.log2(pwm[pwm > 0])
    entropy = np.sum(entropy, axis=1)
    conservation = 2 - entropy

    height_base = 0.0
    logo_height = 1.0
    
    for j in range(pwm.shape[0]) :
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

            if ii == 0 :
                letterAt(nt, j + 0.5, height_base, nt_prob * logo_height, logo_ax, color=None)
            else :
                prev_prob = np.sum(pwm[j, sort_index[:ii]] * conservation[j]) * logo_height
                letterAt(nt, j + 0.5, height_base + prev_prob, nt_prob * logo_height, logo_ax, color=None)

    plt.sca(logo_ax)
    plt.xlim((0, pwm.shape[0]))
    plt.ylim((0, 2))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.axis('off')
    logo_ax.axhline(y=0.01 + height_base, color='black', linestyle='-', linewidth=2)
    
    #Make correlation map
    
    corr_ax.imshow(r_vals[k].T, cmap='RdBu_r', aspect='auto', vmin=-np.max(np.abs(r_vals[k])), vmax=np.max(np.abs(r_vals[k])))
    
    min_max_pos = min(r_vals[k].shape[0], r_vals[k].shape[1])
    corr_ax.plot([0, min_max_pos], [0, min_max_pos], linestyle='--', linewidth=2, color='black')

    plt.sca(corr_ax)
    plt.xlim((0, r_vals[k].shape[0]))
    plt.ylim((0, r_vals[k].shape[1]))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel('Motif (+6 to +' + str(r_vals[k].shape[0]) + ')', fontsize=14)
    plt.ylabel('Cut (+6 to +' + str(r_vals[k].shape[1]) + ')', fontsize=14)
    plt.title('From End of CSE', fontsize=14)
    #plt.axis('off')

def plot_pwm_logo(pwm, figsize=(6, 4)) :
    f = plt.figure(figsize=figsize)
    
    #Make sequence logo
    entropy = np.zeros(pwm.shape)
    entropy[pwm > 0] = pwm[pwm > 0] * -np.log2(pwm[pwm > 0])
    entropy = np.sum(entropy, axis=1)
    conservation = 2 - entropy

    height_base = 0.0
    logo_height = 1.0
    
    logo_ax = plt.gca()
    
    for j in range(pwm.shape[0]) :
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

            if ii == 0 :
                letterAt(nt, j + 0.5, height_base, nt_prob * logo_height, logo_ax, color=None)
            else :
                prev_prob = np.sum(pwm[j, sort_index[:ii]] * conservation[j]) * logo_height
                letterAt(nt, j + 0.5, height_base + prev_prob, nt_prob * logo_height, logo_ax, color=None)

    plt.xlim((0, pwm.shape[0]))
    plt.ylim((0, 2))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.axis('off')
    logo_ax.axhline(y=0.01 + height_base, color='black', linestyle='-', linewidth=2)
    
    plt.tight_layout()
    plt.show()


def get_nmers(n_mer_len) :

    if n_mer_len == 0 :
        return []

    if n_mer_len == 1 :
        return ['A', 'C', 'G', 'T']

    n_mers = []

    prev_n_mers = get_nmers(n_mer_len - 1)

    for _, prev_n_mer in enumerate(prev_n_mers) :
        for _, nt in enumerate(['A', 'C', 'G', 'T']) :
            n_mers.append(prev_n_mer + nt)

    return n_mers

def plot_pwm_logprob_motifs(pwm, n=10, seq_start=0, seq_end=8, figsize=(4, 8)) :
    pwm = pwm[seq_start:seq_end, :]
    
    #Get nmers
    nmers = np.array(get_nmers(pwm.shape[0]), dtype=np.object)
    
    #Score nmers
    nmer_scores = np.zeros(nmers.shape[0])
    for i in range(nmers.shape[0]) :
        score = 0
        nmer = nmers[i]
        for j in range(len(nmer)) :
            if nmer[j] == 'A' :
                score += np.log(pwm[j, 0])
            elif nmer[j] == 'C' :
                score += np.log(pwm[j, 1])
            elif nmer[j] == 'G' :
                score += np.log(pwm[j, 2])
            elif nmer[j] == 'T' :
                score += np.log(pwm[j, 3])
        nmer_scores[i] = score
    
    #Sort scores and pick top selection
    sort_index = np.argsort(nmer_scores)[::-1][:n]
    nmers = nmers[sort_index]
    nmer_scores = nmer_scores[sort_index]
    
    f = plt.figure(figsize=figsize)
    
    plt.imshow(nmer_scores.reshape(-1, 1), cmap='Reds', vmin=np.min(nmer_scores), vmax=np.max(nmer_scores))
    plt.xticks([], [])
    plt.yticks(np.arange(nmer_scores.shape[0]), [nmer + ' ' + str(round(score, 2)) for nmer, score in zip(nmers, nmer_scores)], fontsize=14)
    
    plt.tight_layout()
    plt.show()

