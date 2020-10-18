# COSINE
This repo contains our code for paper [Fine-Tuning Pre-trained Language Model with Weak Supervision: A Contrastive-Regularized Self-Training Approach](https://arxiv.org/abs/2010.07835) (arXiv preprint 2010.07835).

## Model Framework

![BOND-Framework](docs/COSINE.png)

## Benchmark
The results on different datasets are summarized as follows:

| Method | AGNews | IMDB | Yelp | MIT-R | TREC | Chemprot | WiC (dev) |
| ------ | ------- | ----- | ----------- | ------- | -------- | -------- | -------- | 
| Full Supervision (Roberta-base)  | 91.41 | 94.26 | 97.27 |  88.51 | 96.68 | 79.65 |  70.53 |
| Direct Fine-tune with Weak Supervision (Roberta-base) | 82.25 | 72.60 | 74.89 |  70.95 |  62.25 |  44.80 | 59.36 |
| Previous SOTA | 86.28 | 86.98 | 92.05 | 74.41 | 80.20 | 53.48 | 64.88 |
| COSINE | 87.52 | 90.54 | 95.97 | 76.61 | 82.59 | 54.36 | 67.71 | 

- *Previous SOTA*: Self-ensemble/FreeLB/Mixup/SMART (Fine-tuning Approach); Snorkel/WeSTClass/ImplyLoss/Denoise/UST (Weakly-supervised Approach).


## Data

The weakly labeled datasets we used in our experiments are in here: [dataset](data). The statistics of dataset is summarized as follows:

| Dataset | AGNews | IMDB | Yelp | TREC | MIT-R | Chemprot | WiC (dev) |
| ------ | ------- | ----- | ----------- | ------- | -------- | -------- | -------- | 
| Type | Topic | Sentiment | Sentiment |  Slot Filling | Question | Relation |  Word Sense Disambiguation |
| # of Training Samples  | 96k | 20k |  30.4k |  6.6k | 4.8k | 12.6k |  5.4k |
| # of Validation Samples | 12k | 2.5k | 3.8k | 1.0k |  0.6k |    1.6k | 0.6k |
|# of Test Samples | 12k | 2.5k | 3.8k |  1.5k |  0.6k |  1.6k | 1.4k |
| Coverage | 56.4\%  |  87.5\% | 82.8\% | 13.5\% | 95.0\% | 85.9\%  | 63.4\%  |
| Accuracy | 83.1\% | 74.5\% | 71.5\% | 80.7\% | 63.8\% | 46.5\%  | 58.8\%  | 

AGNews &Topic &4 &96k & 12k & 12k & 56.4 & 83.1 \\ %\hline
            IMDB &Sentiment &2 &20k & 2.5k & 2.5k & 87.5 & 74.5 \\ %\hline
            Yelp &Sentiment& 2&30.4k & 3.8k & 3.8k & 82.8 & 71.5 \\ %\hline
            MIT-R & Slot Filling& 9& 6.6k & 1.0k & 1.5k & 13.5 & 80.7 \\ %\hline
            TREC & Question & 6& 4.8k & 0.6k & 0.6k & 95.0  & 63.8 \\ %\hline
            Chemprot & Relation & 10 & 12.6k & 1.6k & 1.6k &  85.9 & 46.5  \\
            WiC & WSD & 2 & 5.4k & 0.6k & 1.4k & 63.4 & 58.8 \\
# Code
- `main.py`: the main code to run the self-training code.

- `dataloader.py`: the code to preprocess text data and tokenize it.

- `utils.py`: some code including calculating accuracy, saving data etc.

- `modeling_roberta.py`: the code to modify the basic Roberta model for our task (we need to directly output the feature vector for RoBERTa)

- `model.py`: the RoBERTa model for text classfication. See `BERT_model` for details.
  
- `trainer.py`: the code to training the RoBERTa under different settings.
   - `train(self)`: training for stage 1
   - `self_adaptive_train(self)`: model using [self-adaptive learning](https://arxiv.org/abs/2002.10319) during self-training (not useful).
   - `selftrain_curriculum(self)`: model using curriculum learning with different proportion of the data during training.
   - `selftrain(self, soft = True, adv = False)`: the code for self-training based on pseudo-labeling with period update. It is similar to previous BOND model for NER.
   - `soft_frequency`: the function to reweight the value of teacher network's prediction based on [WESTClass](https://arxiv.org/abs/1812.11270).
   - `calc_loss`: Calculate the prediction loss for self-training with [confidence-based reweighting](https://arxiv.org/abs/1908.09822).
   - `graph_loss`: Smooth neighbor loss based on [SNTG](https://arxiv.org/abs/1711.00258).
   - `mt_train`: Implemention of [Mean-Teacher](https://arxiv.org/abs/1703.01780) Model.
   - `vat_train`: Implemention of [VAT](https://arxiv.org/abs/1704.03976) Model.
   - `mixup_train`: Implemention of [MixUp](https://openreview.net/forum?id=r1Ddp1-Rb) Model.
   - **Note**: VAT and MT model is based on the implementation in the previous BOND repo.
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
