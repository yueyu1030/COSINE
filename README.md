# COSINE
This repo contains our code for paper [Fine-Tuning Pre-trained Language Model with Weak Supervision: A Contrastive-Regularized Self-Training Approach](https://arxiv.org/abs/2010.07835) (arXiv preprint 2010.07835).

## Model Framework

![BOND-Framework](docs/COSINE.png)

## Benchmark
The results on different datasets are summarized as follows:

| Method | AGNews | IMDB | Yelp | TREC | MIT-R | Chemprot | WiC (dev) |
| ------ | ------- | ----- | ----------- | ------- | -------- | -------- | -------- | 
| Full Supervision (Roberta-base)  | 91.41 | 94.26 | 97.27 |  88.51 | 96.68 | 79.65 |  70.53 |
| Direct Fine-tune with Weak Supervision (Roberta-base) | 82.25 | 72.60 | 74.89 |  70.95 |  62.25 |  44.80 | 59.36 |
| Previous SOTA | 86.28 | 86.98 | 92.05 | 74.41 | 80.20 | 53.48 | 64.88 |
| COSINE | 87.52 | 90.54 | 95.97 | 76.61 | 82.59 | 54.36 | 67.71 | 

- *Previous SOTA*: Self-ensemble/FreeLB/Mixup/SMART (Fine-tuning Approach); Snorkel/WeSTClass/ImplyLoss/Denoise/UST (Weakly-supervised Approach).


## Data

The weakly labeled datasets we used in our experiments are in here: [dataset](dataset)

## Training & Evaluation

We provides the training scripts for all five open-domain distantly/weakly labeled NER datasets in [scripts](scripts). E.g., for BOND training and evaluation on CoNLL03
```
cd BOND
./scripts/conll_self_training.sh
```
For Stage I training and evaluation on CoNLL03
```
cd BOND
./scripts/conll_baseline.sh
```

## Citation

Please cite the following paper if you are using our datasets/tool. Thanks!

```
@article{yu2020finetuning,
  title={Fine-Tuning Pre-trained Language Model with Weak Supervision: A Contrastive-Regularized Self-Training Approach},
  author={Yu, Yue and Zuo, Simiao and Jiang, Haoming and Ren, Wendi and Zhao, Tuo and Zhang, Chao},
  journal   = {CoRR},
  volume    = {abs/2010.07835},
  year={2020},
  url       = {http://arxiv.org/abs/2010.07835},
  archivePrefix = {arXiv},
}
```
