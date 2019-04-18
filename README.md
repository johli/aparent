# APAReNt - APA Regression Net
This repository contains the code for training and running APAReNt, a deep neural network that can predict human 3' UTR Alternative Polyadenylation (APA), annotate genetic variants based on the impact of APA regulation, and engineer new polyadenylation signals according to target isoform abundances or cleavage profiles.

APAReNt was trained on >3.5 million randomized 3' UTR polyadenylation signals expressed on mini gene reporters.

Forward-engineering of new polyadenylation signals is done using the included SeqProp (Stochastic Sequence Backpropagation) software, which implements a gradient-based input maximization algorithm and uses APAReNt as the predictor.
