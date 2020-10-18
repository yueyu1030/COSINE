import pandas as pd
import numpy as np
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis

ABSTAIN = -1
### Keyword based labeling functions ###

## Part of
@labeling_function()
def lf_amino_acid(x):
    return 1 if 'amino acid' in x.sentence.lower() else ABSTAIN

@labeling_function()
def lf_replace(x):
    return 1 if 'replace' in x.sentence.lower() else ABSTAIN

@labeling_function()
def lf_mutant(x):
    return 1 if 'mutant' in x.sentence.lower() or 'mutat' in x.sentence.lower() else ABSTAIN


## Regulator
@labeling_function()
def lf_bind(x):
    return 2 if 'bind' in x.sentence.lower() else ABSTAIN

@labeling_function()
def lf_interact(x):
    return 2 if 'interact' in x.sentence.lower() else ABSTAIN

@labeling_function()
def lf_affinity(x):
    return 2 if 'affinit' in x.sentence.lower() else ABSTAIN


## Upregulator
# Activator
@labeling_function()
def lf_activate(x):
    return 3 if 'activat' in x.sentence.lower() else ABSTAIN

@labeling_function()
def lf_increase(x):
    return 3 if 'increas' in x.sentence.lower() else ABSTAIN

@labeling_function()
def lf_induce(x):
    return 3 if 'induc' in x.sentence.lower() else ABSTAIN

@labeling_function()
def lf_stimulate(x):
    return 3 if 'stimulat' in x.sentence.lower() else ABSTAIN

@labeling_function()
def lf_upregulate(x):
    return 3 if 'upregulat' in x.sentence.lower() else ABSTAIN


## Downregulator
@labeling_function()
def lf_downregulate(x):
    return 4 if 'downregulat' in x.sentence.lower() or 'down-regulat' in x.sentence.lower() else ABSTAIN

@labeling_function()
def lf_reduce(x):
    return 4 if 'reduc' in x.sentence.lower() else ABSTAIN

@labeling_function()
def lf_inhibit(x):
    return 4 if 'inhibit' in x.sentence.lower() else ABSTAIN

@labeling_function()
def lf_decrease(x):
    return 4 if 'decreas' in x.sentence.lower() else ABSTAIN



## Agonist
@labeling_function()
def lf_agonist(x):
    return 5 if ' agoni' in x.sentence.lower() or "\tagoni" in x.sentence.lower() else ABSTAIN


## Antagonist
@labeling_function()
def lf_antagonist(x):
    return 6 if 'antagon' in x.sentence.lower() else ABSTAIN


## Modulator
@labeling_function()
def lf_modulate(x):
    return 7 if 'modulat' in x.sentence.lower() else ABSTAIN

@labeling_function()
def lf_allosteric(x):
    return 7 if 'allosteric' in x.sentence.lower() else ABSTAIN

## Cofactor
@labeling_function()
def lf_cofactor(x):
    return 8 if 'cofactor' in x.sentence.lower() else ABSTAIN


## Substrate/Product
@labeling_function()
def lf_substrate(x):
    return 9 if 'substrate' in x.sentence.lower() else ABSTAIN

@labeling_function()
def lf_transport(x):
    return 9 if 'transport' in x.sentence.lower() else ABSTAIN

@labeling_function()
def lf_catalyze(x):
    return 9 if 'catalyz' in x.sentence.lower() or 'catalys' in x.sentence.lower() else ABSTAIN

@labeling_function()
def lf_product(x):
    return 9 if "produc" in x.sentence.lower() else ABSTAIN

@labeling_function()
def lf_convert(x):
    return 9 if "conver" in x.sentence.lower() else ABSTAIN



## NOT
@labeling_function()
def lf_not(x):
    return 10 if 'not' in x.sentence.lower() else ABSTAIN


