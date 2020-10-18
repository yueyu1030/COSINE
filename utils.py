import os
import random
import logging
import torch.nn as nn
import torch
import numpy as np
from transformers import BertTokenizer, BertConfig, AlbertConfig, AlbertTokenizer, RobertaConfig, RobertaTokenizer

from official_eval import official_f1
from model import RBERT, BERT_model, WiCBERT, ReBERT
from sklearn.metrics import recall_score, precision_recall_fscore_support


MODEL_CLASSES = {
    'bert': (BertConfig, BERT_model, BertTokenizer),
    'roberta': (RobertaConfig, BERT_model, RobertaTokenizer),
    'albert': (AlbertConfig, BERT_model, AlbertTokenizer)
}

MODEL_PATH_MAP = {
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'albert': 'albert-xxlarge-v1'
}

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]

WiCMODEL_CLASSES = {
    'bert': (BertConfig, WiCBERT, BertTokenizer),
    'roberta': (RobertaConfig, WiCBERT, RobertaTokenizer),
    'albert': (AlbertConfig, WiCBERT, AlbertTokenizer)
}
ReMODEL_CLASSES = {
    'bert': (BertConfig, ReBERT, BertTokenizer),
    'roberta': (RobertaConfig, ReBERT, RobertaTokenizer),
    'albert': (AlbertConfig, ReBERT, AlbertTokenizer)
}


def get_label(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.label_file), 'r', encoding='utf-8')]


def load_tokenizer(args):
    tokenizer = MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


def write_prediction_re(args, output_file, preds):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    relation_labels = get_label(args)
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(8001 + idx, relation_labels[pred]))

def write_prediction_tc(args, output_file, preds, id2label):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    #relation_labels = get_label(args)
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, pred in enumerate(preds):
            pred = str(pred)
            pred = int(pred)
            f.write("{}\t{}\n".format(8001 + idx, id2label[pred]))

def write_prediction_re(args, output_file, preds):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    relation_labels = get_label(args)
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(8001 + idx, relation_labels[pred]))


def write_prediction_wic(args, output_file, preds, id2label):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    #relation_labels = get_label(args)
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, pred in enumerate(preds):
            _e = {}
            _e['idx'] = idx
            _e['label'] = 'true' if pred==1 else 'false'
            f.write("{}\n".format(json.dumps(_e)))

def write_f1_tc(args, output_file, f1_macro, f1_micro, acc, global_step):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    #relation_labels = get_label(args)
    import time
    with open(output_file+'_%d.txt'%(args.rule), 'a+', encoding='utf-8') as f:
        #for idx, pred in enumerate(preds):
        basic_str = 'decay=%.1e Global Step:%d Macro_F1:%.4f, Micro_F1:%.4f\n'%(args.weight_decay, global_step, f1_macro, f1_micro)
        if args.method == 'mt':
            pred = '%s beta:%.1e '%(time.ctime(),args.mt_beta) + basic_str
        elif args.method == 'vat':
            pred = '%s eps:%.1e '%(time.ctime(), args.vat_eps) + basic_str
        elif args.method == 'mixup':
            pred = '%s alpha:%.2f '%(time.ctime(), args.mixup_alpha) + basic_str
        elif args.method in ['bond', 'ust']:
            if args.self_training_addvat and args.self_training_addmt:
                pred = '%s vat_mt e:%.1e_b:%.1e eps:%.2f reg:%.2f g:%.1f cyc:%d '%(time.ctime(),args.vat_eps, args.mt_beta, args.bond_eps, args.self_training_reg, args.self_training_graph_loss, args.self_training_update_period) + basic_str
            elif args.self_training_addvat:
                assert args.self_training_addmt == 0
                pred = '%s vat_e:%.1e eps:%.2f reg:%.2f g:%.1f cyc:%d '%(time.ctime(), args.vat_eps, args.bond_eps, args.self_training_reg, args.self_training_graph_loss, args.self_training_update_period) + basic_str
            elif args.self_training_addmt:
                assert args.self_training_addvat == 0
                pred = '%s mt_b:%.1e eps:%.2f reg:%.2f g:%.1f cyc:%d '%(time.ctime(), args.mt_beta, args.bond_eps, args.self_training_reg, args.self_training_graph_loss, args.self_training_update_period) + basic_str
            else:
                assert args.self_training_addmt == 0 and args.self_training_addmt == 0
                pred = '%s eps:%.2f reg:%.2f g:%.1f cyc:%d '%(time.ctime(), args.bond_eps, args.self_training_reg, args.self_training_graph_loss, args.self_training_update_period) + basic_str
        elif args.method in ['curr']:
            if args.self_training_addvat and args.self_training_addmt:
                pred = '%s vat_mt e:%.1e_b:%.1e eps:%.2f reg:%.2f g:%.1f cyc:%d '%(time.ctime(),args.vat_eps, args.mt_beta, args.bond_eps, args.self_training_reg, args.self_training_graph_loss, args.self_training_update_period) + basic_str
            elif args.self_training_addvat:
                assert args.self_training_addmt == 0
                pred = '%s vat_e:%.1e eps:%.2f reg:%.2f g:%.1f cyc:%d '%(time.ctime(), args.vat_eps, args.bond_eps, args.self_training_reg, args.self_training_graph_loss, args.self_training_update_period) + basic_str
            elif args.self_training_addmt:
                assert args.self_training_addvat == 0
                pred = '%s mt_b:%.1e eps:%.2f reg:%.2f g:%.1f cyc:%d '%(time.ctime(), args.mt_beta, args.bond_eps, args.self_training_reg, args.self_training_graph_loss, args.self_training_update_period) + basic_str
            else:
                assert args.self_training_addmt == 0 and args.self_training_addmt == 0
                pred = '%s eps:%.2f reg:%.2f g:%.1f cyc:%d maxratio:%.1f'%(time.ctime(), args.bond_eps, args.self_training_reg, args.self_training_graph_loss, args.self_training_update_period, args.curr_max_ratio) + basic_str

        elif args.method == 'sat':
            pred = '%s es:%d mom:%.1f a:%.1f b:%.1f '%(time.ctime(), args.sat_es, args.sat_momentum, args.sat_alpha, args.sat_beta) + basic_str
        else: 
            pred = '%s w:%.1f '%(time.ctime(), args.soft_label_weight) + basic_str

        f.write(pred)

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0, metric = 'l2'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.metric = metric
        # print('ContrastiveLoss, Metric:', self.metric)

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        #elf.check_type_forward((x0, x1, y))

        # euclidian distance
        if self.metric == 'l2':
            diff = x0 - x1
            dist_sq = torch.sum(torch.pow(diff, 2), 1) / x0.shape[-1]
            dist = torch.sqrt(dist_sq)
        elif self.metric == 'cos':
            prod = torch.sum(x0 * x1, -1)
            dist = 1 - prod /  torch.sqrt(torch.sum(x0**2, 1) * torch.sum(x1**2, 1))
            dist_sq = dist ** 2
            #print(x0, x1, torch.sum(torch.pow(x0-x1, 2), 1) / x0.shape[-1], dist, dist_sq)
        else:
            print("Error Loss Metric!!")
            return 0
        #dist = torch.sum( - x0 * x1 / np.sqrt(x0.shape[-1]), 1).exp()
        #dist_sq = dist ** 2

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss, dist_sq, dist

class SoftContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0, metric = 'l2'):
        super(SoftContrastiveLoss, self).__init__()
        self.margin = margin
        self.metric = metric

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y, margin):
        #elf.check_type_forward((x0, x1, y))

        # euclidian distance
        if self.metric == 'l2':
            diff = x0 - x1
            dist_sq = torch.sum(torch.pow(diff, 2), 1) / x0.shape[-1]
            dist = torch.sqrt(dist_sq)
        elif self.metric == 'cos':
            prod = torch.sum(x0 * x1, -1)
            dist = 1 - prod /  torch.sqrt(torch.sum(x0**2, 1) * torch.sum(x1**2, 1))
            dist_sq = dist ** 2
        # diff = x0 - x1
        # dist_sq = torch.sum(torch.pow(diff, 2), 1) / x0.shape[-1]
        # dist = torch.sqrt(dist_sq)
        d_pos = dist - margin
        mdist = margin - dist
        dist_pos = torch.clamp(d_pos, min = 0.0)
        dist_neg = torch.clamp(mdist, min = 0.0)
        loss = y * torch.pow(dist_pos, 2) + (1 - y) * torch.pow(dist_neg, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss, dist_sq, dist

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        # print('yes')
        # assert 0
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, average='macro'):
    acc = simple_accuracy(preds, labels)
    #macro_recall = recall_score(y_true=labels, y_pred = preds, average = 'macro')
    #micro_recall = recall_score(y_true=labels, y_pred = preds, average = 'micro')
    #print(acc, macro_recall, micro_recall)
    pr, re, f1cat, _ = precision_recall_fscore_support(y_true = labels, y_pred = preds, average=None)
    #print(pr, re, f1cat)
    pr, re, f1, _ = precision_recall_fscore_support(y_true = labels, y_pred = preds, average='macro')
    #print(pr, re, f1)
    p, r, f, _ = precision_recall_fscore_support(y_true = labels, y_pred = preds, average='micro')
    #print(labels[:10], preds[:10])
    return {
        "acc": acc,
        #"f1": official_f1(),
        "recall": re,
        "macro-f1": f1,
        "micro-f1": f,
        "f1-cat": f1cat,
    }

