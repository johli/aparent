# APARENT - APA Regression Net
This repository contains the code for training and running APARENT, a deep neural network that can predict human 3' UTR Alternative Polyadenylation (APA), annotate genetic variants based on the impact of APA regulation, and engineer new polyadenylation signals according to target isoform abundances or cleavage profiles.

APARENT was trained on >3.5 million randomized 3' UTR polyadenylation signals expressed on mini gene reporters in HEK293 cells.

Forward-engineering of new polyadenylation signals is done using the included SeqProp (Stochastic Sequence Backpropagation) software, which implements a gradient-based input optimization algorithm and uses APARENT as the predictor.

### Installation
APARENT can be installed by cloning or forking the [github repository](https://github.com/johli/aparent.git):
```sh
git clone https://github.com/johli/aparent.git
cd aparent
python setup.py install
```

#### APARENT requires the following packages to be installed
- Tensorflow >= 1.13.1
- Keras >= 2.2.4
- Scipy >= 1.2.1
- Numpy >= 1.16.2
- Isolearn >= 0.2.0 ([github](https://github.com/johli/isolearn.git))
- [Optional] SeqProp >= 0.1 ([github](https://github.com/johli/seqprop.git))

### Usage
APARENT is built as a Keras Model, and as such can be easily loaded and used to predict APA using Keras function calls.
Please see the example usage notebooks provided below for an in-depth tutorial on how to use the model for APA- and Variant Effect prediction.

This simple example illustrates how to predict the isoform abundance and cleavage profile of an input APA event:
```python
import keras
from keras.models import Sequential, Model, load_model
from aparent.predictor import *

#Load APADB-tuned APARENT model and input encoder
apadb_model = load_model('../saved_models/aparent_apadb_fitted_large_lessdropout_no_sampleweights.h5')
apadb_encoder = get_apadb_encoder()

#Example APA sites (gene = PSMC6)

#Proximal and Distal PAS Sequences
seq_prox = 'AGATAGTGGTATAAGAAAGCATTTCTTATGACTTATTTTGTATCATTTGTTTTCCTCATCTAAAAAGTTGAATAAAATCTGTTTGATTCAGTTCTCCTACATATATATTCTTGTCTTTTCTGAGTATATTTACTGTGGTCCTTTAGGTTCTTTAGCAAGTAAACTATTTGATAACCCAGATGGATTGTGGATTTTTGAATATTAT'
seq_dist = 'TGGATTGTGGATTTTTGAATATTATTTTAAAATAGTACACATACTTAATGTTCATAAGATCATCTTCTTAAATAAAACATGGATGTGTGGGTATGTCTGTACTCCTCCTTTCAGAAAGTGTTTACATATTCTTCATCTACTGTGATTAAGCTCATTGTTGGTTAATTGAAAATATACATGCACATCCATAACTTTTTAAAGAGTA'

#Site Distance
site_distance = 180

#Proximal and Distal cut intervals within each sequence defining the isoforms
prox_cut_start, prox_cut_end = 80, 105
dist_cut_start, dist_cut_end = 80, 105

#Predict with APADB-tuned APARENT model
iso_pred, cut_prox, cut_dist = apadb_model.predict(x=apadb_encoder([seq_prox], [seq_dist], [prox_cut_start], [prox_cut_end], [dist_cut_start], [dist_cut_end], [site_distance]))

print("Predicted proximal vs. distal isoform % (APADB) = " + str(iso_pred[0, 0]))
```

## APARENT Example Usage Notebooks
[Notebook 1: APA Isoform & Cleavage Prediction](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/examples/aparent_example_isoform_prediction.ipynb)<br/>
[Notebook 2: APA Variant Effect Prediction](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/examples/aparent_example_variant_prediction.ipynb)<br/>

## Legacy Model & Code Availability
[Legacy Code Repository](https://github.com/johli/aparent-legacy)<br/>

## Data Availability
[Processed Data Repository](https://drive.google.com/open?id=1qex3oY-rarsd7YowM7TxxUklLbLkUyOT)<br/>
[Processed Data Repository (legacy)](https://drive.google.com/open?id=1Q2tTIRIR0C3kL7stI51TPLdGMdbZ0WnV)<br/>

## Random MPRA Linear Model Notebooks
[Notebook 1a: Isoform Log Odds Ratio Analysis (Alien1 Library)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_random_mpra_alien1_isoform_logodds_ratios.ipynb)<br/>
[Notebook 1b: Isoform Log Odds Ratio Analysis (Alien2 Library)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_random_mpra_alien2_isoform_logodds_ratios.ipynb)<br/>
[Notebook 2: Cleavage Log Odds Ratio Analysis (Alien1 Library)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_random_mpra_alien1_cleavage_logodds_ratios.ipynb)<br/>
[Notebook 3a: Hexamer Logistic Regression (Combined Library)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_random_mpra_combined_logistic_regression.ipynb)<br/>
[Notebook 3b: Hexamer Logistic Regression (TOMM5 Library only)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_random_mpra_tomm5_logistic_regression.ipynb)<br/>
[Notebook 3c: Hexamer Logistic Regression (Alien1 Library only)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_random_mpra_alien1_logistic_regression.ipynb)<br/>
[Notebook 3d: Hexamer Logistic Regression (Alien2 Library only)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_random_mpra_alien2_logistic_regression.ipynb)<br/>

## Random MPRA Neural Network Notebooks
[Notebook 1: MPRA Prediction Evaluation](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/evaluate_aparent_random_mpra_legacy.ipynb)<br/>
[Notebook 2a: Conv Layer 1 and 2 Analysis (Alien1 Library)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_aparent_conv_layers_alien1_legacy.ipynb)<br/>
[Notebook 2b: Conv Layer 1 and 2 Analysis (Alien2 Library)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_aparent_conv_layers_alien2_legacy.ipynb)<br/>
[Notebook 3: CSE Hexamer Filter (Conv Layer 1)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_aparent_conv_layer_1_scaled_alien2_legacy.ipynb) <br/>
[Notebook 4: Cleavage Motifs (Conv Layer 1)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_aparent_conv_layer_1_cleavage_alien1_memory_efficient_legacy.ipynb) <br/>

## SeqProp APA Engineering Notebooks
[Notebook 1: Target Isoform Sequence Optimization](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/seqprop/seqprop_aparent_isoform_optimization_legacy.ipynb)<br/>
[Notebook 2: Target Cleavage Sequence Optimization](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/seqprop/seqprop_aparent_cleavage_optimization_legacy.ipynb)<br/>
[Notebook 3: Dense Layer Sequence Visualization (DeepDream-Style)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/seqprop/seqprop_aparent_deepdream_optimization_legacy.ipynb)<br/>

## Designed MPRA Analysis Notebooks
[Notebook 0a: Basic MPRA Library Statistics](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_aparent_designed_mpra_stats_legacy.ipynb)<br/>
[Notebook 0b: MPRA LoFi vs. HiFi Replicates](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_aparent_designed_mpra_lofi_vs_hifi_legacy.ipynb)<br/>

[Notebook 1a: SeqProp Target Isoforms (Summary)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_aparent_designed_mpra_seqprop_iso_summary_legacy.ipynb)<br/>
[Notebook 1b: SeqProp Target Isoforms (Detailed)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_aparent_designed_mpra_seqprop_iso_detailed_legacy.ipynb)<br/>

[Notebook 2a: SeqProp Target Cut (Summary)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_aparent_designed_mpra_seqprop_cut_summary_legacy.ipynb)<br/>
[Notebook 2b: SeqProp Target Cut (Detailed)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_aparent_designed_mpra_seqprop_cut_detailed_legacy.ipynb)<br/>

[Notebook 3: Human Wildtype APA Prediction](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_aparent_designed_mpra_wildtype_human_apa_legacy.ipynb)<br/>

[Notebook 4a: Human Variant Analysis (Summary)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_aparent_designed_mpra_variant_summary_legacy.ipynb)<br/>
[Notebook 4b: Disease-Implicated Variants/UTRs (Detailed)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_aparent_designed_mpra_pathogenic_utrs_legacy.ipynb)<br/>
[Notebook 4c: Cleavage-Altering Variants (Detailed)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_aparent_designed_mpra_complex_cut_variants_legacy.ipynb)<br/>

[Notebook 5a: Complex Functional Variants (Summary)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_aparent_designed_mpra_rare_functional_variants_summary_legacy.ipynb)<br/>
[Notebook 5b: Complex Functional Variants (Canonical CSE)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_aparent_designed_mpra_rare_functional_variants_detailed_canonical_cse_legacy.ipynb)<br/>
[Notebook 5c: Complex Functional Variants (Cryptic CSE)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_aparent_designed_mpra_rare_functional_variants_detailed_cryptic_cse_legacy.ipynb)<br/>
[Notebook 5d: Complex Functional Variants (CFIm25)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_aparent_designed_mpra_rare_functional_variants_detailed_tgta_legacy.ipynb)<br/>
[Notebook 5e: Complex Functional Variants (CstF)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_aparent_designed_mpra_rare_functional_variants_detailed_tgtct_legacy.ipynb)<br/>
[Notebook 5f: Complex Functional Variants (Folding)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_aparent_designed_mpra_rare_functional_variants_detailed_folding_legacy.ipynb)<br/>

[Notebook Bonus: TGTA Motif Saturation Mutagenesis](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_aparent_designed_mpra_tgta_mutation_maps_legacy.ipynb)<br/>

## Native APA Analysis Notebooks
#### (APADB/Leslie APA Atlas)
[Notebook 0: Basic Data Statistics](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_leslie_apadb_celltypes_basic_stats_legacy.ipynb)<br/>
[Notebook 1: Differential Usage Analysis](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_leslie_apadb_celltypes_differential_usage_legacy.ipynb)<br/>
[Notebook 2: Cleavage Site Prediction](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_leslie_apadb_celltypes_cleavage_predictions_legacy.ipynb)<br/>
[Notebook 3: APA Isoform Prediction](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_leslie_apadb_celltypes_isoform_predictions_legacy.ipynb)<br/>
[Notebook 4: APA Isoform Prediction (Cross-Validation)](https://nbviewer.jupyter.org/github/johli/aparent/blob/master/analysis/analyze_leslie_apadb_celltypes_crossvalidate_isoform_predictions_legacy.ipynb)<br/>
