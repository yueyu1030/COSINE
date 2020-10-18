import os
import logging
from tqdm import tqdm, trange
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset, TensorDataset
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
import copy
import math
from model import RBERT
from utils import set_seed, write_f1_tc, write_prediction_re, write_prediction_tc, write_prediction_wic, compute_metrics, get_label, MODEL_CLASSES, WiCMODEL_CLASSES, ReMODEL_CLASSES, ContrastiveLoss, SoftContrastiveLoss

logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, args, train_dataset = None, dev_dataset = None, test_dataset = None, labelset = None, unlabeled = None, \
                num_labels = 10, id2label = None, label2id = None, data_size = 100):
                #masked_train_dataset = None, masked_dev_dataset = None,  masked_test_dataset = None, masked_unlabeled_dataset = None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.unlabeled = unlabeled
        self.data_size = data_size

        self.label_lst = labelset
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

        self.w = args.soft_label_weight
        self.k = (1-self.w)/(self.num_labels-1)
        self.label_matrix = torch.eye(self.num_labels) * (self.w - self.k) + self.k * torch.ones(self.num_labels)

        if args.task_type == 'wic':
            self.config_class, self.model_class, _ = WiCMODEL_CLASSES[args.model_type]
        elif args.task_type == 're':
            self.config_class, self.model_class, _ = ReMODEL_CLASSES[args.model_type]
        else:
            self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        self.bert_config = self.config_class.from_pretrained(args.model_name_or_path, num_labels=self.num_labels, finetuning_task=args.task)
        
        self.model = self.model_class(self.bert_config, args)
        self.init_model()
        #self.model.to(self.device)

    def init_model(self):
        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu"
        self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)


    def calc_loss(self, input, target, loss, thresh = 0.95, soft = True, conf = 'max', confreg = 0.1):
        softmax = nn.Softmax(dim=1)
        target = softmax(target.view(-1, target.shape[-1])).view(target.shape)
        
        if conf == 'max':
            weight = torch.max(target, axis = 1).values
            w = torch.FloatTensor([1 if x == True else 0 for x in weight>thresh]).to(self.device)
        elif conf == 'entropy':
            weight = torch.sum(-torch.log(target+1e-6) * target, dim = 1)
            weight = 1 - weight / np.log(weight.size(-1))
            w = torch.FloatTensor([1 if x == True else 0 for x in weight>thresh]).to(self.device)
        target = self.soft_frequency(target, probs = True, soft = soft)
        
        loss_batch = loss(input, target)

        l = torch.sum(loss_batch * w.unsqueeze(1) * weight.unsqueeze(1))
        
        n_classes_ = input.shape[-1]
        l -= confreg *( torch.sum(input * w.unsqueeze(1)) + np.log(n_classes_) * n_classes_ )
        return l

    def graph_loss(self, input, feat, target, conf = 'none', thresh = 0.1, distmetric = 'l2'):
        softmax = nn.Softmax(dim=1)
        target = softmax(target.view(-1, target.shape[-1])).view(target.shape)
        if conf == 'max':
            weight = torch.max(target, axis = 1).values
            w = torch.tensor([i for i,x in enumerate(weight) if x > thresh], dtype=torch.long).to(self.device)
        elif conf == 'entropy':
            weight = torch.sum(-torch.log(target+1e-6) * target, dim = 1)
            weight = 1 - weight / np.log(weight.size(-1))
            w = torch.tensor([i for i,x in enumerate(weight) if x > thresh], dtype=torch.long).to(self.device)
        input_x = input[w]

        feat_x = feat[w]
        batch_size = input_x.size()[0]
        if batch_size == 0:
            return 0
        index = torch.randperm(batch_size).to(self.device)
        input_y = input_x[index, :]
        feat_y = feat_x[index, :]
        argmax_x = torch.argmax(input_x, dim = 1)
        argmax_y = torch.argmax(input_y, dim = 1)
        agreement = torch.FloatTensor([1 if x == True else 0 for x in argmax_x == argmax_y]).to(self.device)

        criterion = ContrastiveLoss(margin = 1.0, metric = distmetric)
        loss, dist_sq, dist = criterion(feat_x, feat_y, agreement)
        
        return loss

    def soft_frequency(self, logits,  probs=False, soft = True):
        """
        Unsupervised Deep Embedding for Clustering Analysis
        https://arxiv.org/abs/1511.06335
        """
        power = self.args.self_training_power
        if not probs:
            softmax = nn.Softmax(dim=1)
            y = softmax(logits.view(-1, logits.shape[-1])).view(logits.shape)
        else:
            y = logits
        f = torch.sum(y, dim=0)
        t = y**power / f
        #print('t', t)
        t = t + 1e-10
        p = t/torch.sum(t, dim=-1, keepdim=True)
        return p if soft else torch.argmax(p, dim=1)

    

    def selftrain(self, soft = True, adv = False):
        selftrain_dataset = ConcatDataset([self.train_dataset, self.unlabeled])
        ## generating pseudo_labels
        pseudo_labels = []
        train_sampler = RandomSampler(selftrain_dataset)
        train_dataloader = DataLoader(selftrain_dataset, sampler=train_sampler, batch_size=self.args.batch_size)
        if self.args.self_training_max_step > 0:
            t_total = self.args.self_training_max_step
            self.args.num_train_epochs = self.args.self_training_max_step // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)
        self_training_loss = nn.KLDivLoss(reduction = 'none') if soft else nn.CrossEntropyLoss(reduction = 'none')
        softmax = nn.Softmax(dim=1)
        update_step = 0
        self_training_steps = self.args.self_training_max_step
        global_step = 0
        selftrain_loss = 0
        set_seed(self.args)
        #self.model.zero_grad()
        for t3 in range(int(self_training_steps/len(train_dataloader)) + 1):
            epoch_iterator = tqdm(train_dataloader, desc="SelfTrain, Iteration")
            for step, batch in enumerate(epoch_iterator):
                if global_step % self.args.self_training_update_period == 0:
                    teacher_model = copy.deepcopy(self.model) #.to("cuda")
                    teacher_model.eval()
                    for p in teacher_model.parameters():
                        p.requires_grad = False
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU     
                inputs = {
                            'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],

                        }
                #self.model.eval()
                if self.args.task_type=='wic':
                    inputs['keys'] = batch[6]
                elif self.args.task_type=='re':
                    inputs['e1_mask'] = batch[4]
                    inputs['e2_mask'] = batch[5]
                outputs = self.model(**inputs)
                outputs_pseudo = teacher_model(**inputs)

                logits = outputs[0]
                true_labels = batch[-1]
                
                loss = self.calc_loss(input = torch.log(softmax(logits)), \
                                        target= outputs_pseudo[0], \
                                        loss = self_training_loss, \
                                        thresh = self.args.bond_eps, \
                                        soft = soft, \
                                        conf = 'entropy', \
                                        confreg = self.args.self_training_confreg)

                if self.args.self_training_contrastive_weight > 0:
                    contrastive_loss = self.graph_loss(input = torch.log(softmax(logits)), \
                                        feat = outputs_pseudo[-1], \
                                        target= outputs_pseudo[0], \
                                        conf = 'entropy', \
                                        thresh =  self.args.bond_eps, \
                                        distmetric = self.args.distmetric, \
                                        )
                    loss = loss + self.args.self_training_contrastive_weight * contrastive_loss

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()
                selftrain_loss += loss.item()
                loss.backward()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    teacher_model.zero_grad()
                    global_step += 1
                    epoch_iterator.set_description("SelfTrain iter:%d Loss:%.3f m:%.3f" % (step, selftrain_loss/global_step, ))
                    if self.args.logging_steps > 0 and global_step % self.args.self_train_logging_steps == 0:
                        self.evaluate('dev', global_step)
                        self.evaluate('test', global_step)

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model()

                if 0 < self.args.self_training_max_step < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.self_training_max_step < global_step:
                break
        pass  


    def train(self):
        if self.args.method == 'clean':
            print('clean data!')
            concatdataset = ConcatDataset([self.train_dataset, self.unlabeled])
            train_sampler = RandomSampler(concatdataset)
            train_dataloader = DataLoader(concatdataset, sampler=train_sampler, batch_size = self.args.batch_size)
        else:
            train_sampler = RandomSampler(self.train_dataset)
            train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)
        #assert 0
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        set_seed(self.args)
        criterion = nn.KLDivLoss(reduction = 'batchmean')

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {
                            'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels': batch[3],
                          }
                if self.args.task_type=='wic':
                    inputs['keys'] = batch[6]
                elif self.args.task_type=='re':
                    inputs['e1_mask'] = batch[4]
                    inputs['e2_mask'] = batch[5]
                outputs = self.model(**inputs)
                loss1 = outputs[0]
                logits = outputs[1]
                loss = criterion(input = F.log_softmax(logits), target = self.label_matrix[batch[3]].to(self.device))
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                if torch.cuda.device_count() > 1:
                    #print(loss.size(), torch.cuda.device_count())
                    loss = loss.mean()
                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1
                    epoch_iterator.set_description("iteration:%d, w=%.1f, Loss:%.3f" % (_, self.args.soft_label_weight, tr_loss/global_step))
                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        self.evaluate('dev', global_step)
                        self.evaluate('test', global_step)
                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model()
                
                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break
        #assert 0
        return global_step, tr_loss / global_step

    def evaluate(self, mode, global_step=-1):
        # We use test dataset because semeval doesn't have dev dataset
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                            'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels': batch[3],
                          }
                if self.args.task_type=='wic':
                    inputs['keys'] = batch[6]
                elif self.args.task_type=='re':
                    inputs['e1_mask'] = batch[4]
                    inputs['e2_mask'] = batch[5]
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }
        preds = np.argmax(preds, axis=1)
        if self.args.task_type == 're':
            write_prediction_re(self.args, os.path.join(self.args.eval_dir, "pred/proposed_answers.txt"), preds)
        elif self.args.task_type == 'tc':
            write_prediction_tc(self.args, os.path.join(self.args.eval_dir, "pred/pred_%s_%s_%s_%d.txt"%(self.args.task, mode, self.args.method, global_step)), preds, self.id2label)
        elif self.args.task_type == 'wic':
            write_prediction_wic(self.args, os.path.join(self.args.eval_dir, "pred/pred_%s_%s_%s_%s.txt"%(self.args.task, mode, self.args.method, str(global_step))), preds, self.id2label)
        else:
            pass
        result = compute_metrics(preds, out_label_ids)
        result.update(result)

        logger.info("***** Eval results *****")

        print('Macro F1: %.4f, Micro F1: %.4f, Accu: %.4f'%(result["macro-f1"], result["micro-f1"], result["acc"]))
        write_f1_tc(self.args, os.path.join(self.args.eval_dir, "pred_%s_%s"%(self.args.task, mode)), result["macro-f1"], result["micro-f1"], result["acc"],global_step)
  
        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        output_dir = os.path.join(self.args.model_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_config.bin'))
        logger.info("Saving model checkpoint to %s", output_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.bert_config = self.config_class.from_pretrained(self.args.model_dir)
            logger.info("***** Config loaded *****")
            self.model = self.model_class.from_pretrained(self.args.model_dir, config=self.bert_config, args=self.args)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
         