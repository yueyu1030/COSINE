import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, RobertaModel, AlbertModel
from transformers import (
    BertForSequenceClassification,
    RobertaModel,
    AlbertModel,
    BertModel,
    BertForTokenClassification,
    RobertaForSequenceClassification,
    BertForMaskedLM,
    RobertaForMaskedLM,
    AlbertForMaskedLM
    )
from modeling_roberta import RobertaForSequenceClassification_v2
from transformers.modeling_roberta import RobertaLMHead
from transformers.modeling_bert import BertOnlyMLMHead
from transformers.modeling_albert import AlbertMLMHead

PRETRAINED_MODEL_MAP = {
    'bert': BertModel,
    'roberta': RobertaModel,
    'albert': AlbertModel
}

PRETRAINED_MODEL_LM_MAP = {
    'bert': BertOnlyMLMHead,
    'roberta': RobertaLMHead,
    'albert': AlbertMLMHead
}

PRETRAINED_MODEL_MAP_SeqClass = {
    'bert': BertForSequenceClassification,
    'roberta': RobertaForSequenceClassification_v2,
    'albert': AlbertModel
}

PRETRAINED_MODEL_MAP_TokenClass = {
    'bert': BertForTokenClassification,
    'roberta': RobertaModel,
    'albert': AlbertModel
}

class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class BERT_model(BertPreTrainedModel):
    def __init__(self, bert_config, args):
        super(BERT_model, self).__init__(bert_config)
        if args.task_type == 're' or 'tc':
            self.bert = PRETRAINED_MODEL_MAP_SeqClass[args.model_type].from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert
        else:
            self.bert = PRETRAINED_MODEL_MAP_TokenClass[args.model_type].from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert
        #self.fc_layer = FCLayer(bert_config.hidden_size, bert_config.num_labels, args.dropout_rate, use_activation=False)
        #self.lm_head = RobertaLMHead(config = bert_config)
        self.args = args

    def forward(self, input_ids, attention_mask, token_type_ids, inputs_embeds = None, labels = None, e1_mask = None, e2_mask = None):
        #print(labels)
        if input_ids is None:
            outputs = self.bert(inputs_embeds = inputs_embeds, attention_mask = attention_mask,
                                token_type_ids = token_type_ids, labels = labels)  # sequence_output, pooled_output, (hidden_states), (attentions)
            #sequence_output = outputs[0]

        elif labels is not None:
            outputs = self.bert(input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids, labels = labels)  # sequence_output, pooled_output, (hidden_states), (attentions)
            #sequence_output = outputs[0]
            #pooled_output = outputs[1]  # [CLS]
            if self.args.task_type == 're' or 'tc':
                '''
                loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
                Classification (or regression if config.num_labels==1) loss.
                logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
                Classification (or regression if config.num_labels==1) scores (before SoftMax).
                '''
                loss, logits = outputs[:2]
            else:
                '''
                loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided)
                Classification loss.
                scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
                '''
                loss, scores = outputs[:2]
        else:
            outputs = self.bert(input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
            #sequence_output = outputs[0]
            #pooled_output = outputs[1]  # [CLS]
            if self.args.task_type == 're' or 'tc':
                logits = outputs[0]
            else:
                scores = outputs[0]
        return outputs
            #logits = self.fc_layer(sequence_output)

    def forward_pretrain(self, input_ids, attention_mask, masked_lm_labels):
        out = self.bert.roberta(input_ids, attention_mask)
        sequence_output = out[0]
        prediction_scores = self.lm_head(sequence_output)

        outputs = (prediction_scores,) + out[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs
        return outputs


class RBERT(BertPreTrainedModel):
    def __init__(self, bert_config, args):
        super(RBERT, self).__init__(bert_config)
        self.bert = PRETRAINED_MODEL_MAP[args.model_type].from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert
        class pseudoclass(object):
            def __init__(self, base):
                self.base = base
            def get_input_embeddings(self):
                return self.base.get_input_embeddings()
        self.bert.roberta = pseudoclass(self.bert)
        self.extended_token_type_embeddings = nn.Embedding(bert_config.type_vocab_size+20, bert_config.hidden_size)
        nn.init.zeros_(self.extended_token_type_embeddings.weight)
        for k in range(bert_config.type_vocab_size+20):
            self.extended_token_type_embeddings.weight.data[k,:] =  self.bert.embeddings.token_type_embeddings.weight.data[0,:]
        self.extended_token_type_embeddings.weight.data[:bert_config.type_vocab_size,:] =  self.bert.embeddings.token_type_embeddings.weight.data
        self.extended_token_type_embeddings.weight.data[10:10+bert_config.type_vocab_size,:] =  self.bert.embeddings.token_type_embeddings.weight.data
        self.bert.embeddings.token_type_embeddings = self.extended_token_type_embeddings

        self.num_labels = bert_config.num_labels

        self.cls_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        self.e1_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        self.e2_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        self.label_classifier = FCLayer(bert_config.hidden_size * 3, bert_config.num_labels, args.dropout_rate, use_activation=False)

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask, return_hidden=False, inputs_embeds=None):
        if input_ids is None:
            outputs = self.bert(inputs_embeds = inputs_embeds, attention_mask = attention_mask,
                                token_type_ids = token_type_ids)
        else:
            outputs = self.bert(input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.e1_fc_layer(e1_h)
        e2_h = self.e2_fc_layer(e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        if return_hidden:
            return [concat_h]
        logits = self.label_classifier(concat_h)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class WiCBERT(RBERT):
    def forward(self, input_ids, keys, attention_mask, token_type_ids, labels=None, return_hidden=False, **args):
        if input_ids is not None:
            e1_mask = (keys==1).long()
            e2_mask = (keys==2).long()
            token_type_ids = token_type_ids + (keys>1).long()*10
        return super().forward(input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask, return_hidden=return_hidden, **args)

class ReBERT(RBERT):
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, e1_mask=None, e2_mask=None, return_hidden=False, **args):
        if input_ids is not None:
            token_type_ids = token_type_ids + e1_mask*1 + e2_mask*2
        return super().forward(input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask, return_hidden=return_hidden, **args)
