import os
import csv
import copy
import json
import logging
import random
import torch
from torch.utils.data import TensorDataset

from utils import get_label

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, label, true = -1):
        self.guid = guid
        self.text_a = text_a
        self.label = label
        self.true = true

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class ReInputExample(InputExample):
    def __init__(self, guid, text_a, span_a, span_b, label, true = -1):
        self.guid = guid
        self.text_a = text_a
        self.span_a = span_a
        self.span_b = span_b
        self.label = label
        self.true = true
    
class WiCInputExample(InputExample):
    def __init__(self, guid, text_a, text_b, span_a, span_b, label, true = -1):
        self.guid = guid
        self.text_a = text_a
        self.span_a = span_a
        self.text_b = text_b
        self.span_b = span_b
        self.label = label
        self.true = true



class MaskedLmInstance(object):
    """
    A single set of features of masked data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    """
    def __init__(self, input_ids, attention_mask, masked_token_id, masked_true_label,
                 ):
        self.input_ids = input_ids
        self.masked_token_id = masked_token_id
        self.masked_true_label = masked_true_label
        self.attention_mask = attention_mask

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id, true = -1,
                 e1_mask = None, e2_mask = None, keys=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.true = true
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask
        self.keys=keys

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class SemEvalProcessor(object):
    """Processor for the Semeval data set """

    def __init__(self, args):
        self.args = args
        self.relation_labels = get_label(args)

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = self.relation_labels.index(line[0])
            if i % 1000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._create_examples(self._read_tsv(os.path.join(self.args.data_dir, file_to_read)), mode)

class YelpProcessor(object):
    """Processor for the Yelp data set """

    def __init__(self, args):
        self.args = args
        #self.relation_labels = self.load_json(filename) # all possible labels
        filename = args.data_dir + '/' + 'config.json'
        label, num_label, label2id, id2label = self.load_info(filename)
        self.relation_labels = label
        self.num_label = num_label
        self.label2id = label2id
        self.id2label = id2label

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def read_data(self, filename, mode):
        path = filename + '/' + mode + '_data.json'
        path = filename
        with open(path, 'r') as f:
            data = json.load(f)
        for i in range(len(data)):
            data[i]["labelid"] = self.label2id[data[i]["label"]]
        return data

    def load_info(self, filename):
        with open(filename, 'r') as f:
            file = json.load(f)
        label2id = file["label2id"]
        num_label = file["labels"]
        id2label = file["id2label"]
        label = [id2label[str(int(i))] for i in range(num_label)]
        return label, num_label, label2id, id2label

    def _create_examples(self, data, set_type):
        examples = []
        for i, d in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            text_a = d["text"]
            label = d["labelid"]
            if i % 2000 == 0:
                logger.info(d)
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def _create_examples_raw(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = self.relation_labels.index(line[0])
            if i % 2000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file
        elif mode == 'unlabeled':
            file_to_read = self.args.unlabel_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._create_examples(self.read_data(os.path.join(self.args.data_dir, file_to_read), mode), mode)

class YoutubeProcessor(object):
    """Processor for the Yelp data set """
    def __init__(self, args):
        self.args = args
        #self.relation_labels = self.load_json(filename) # all possible labels
        #filename = args.data_dir + '/' + 'config.json'
        #label, num_label, label2id, id2label = self.load_info(filename)
        #self.relation_labels =
        #self.num_label = num_label
        #self.label2id = None
        #self.id2label = None
        self.rule = self.args.rule
        if 'agnews' in self.args.task:
            self.num_label = 4
        elif self.args.task == 'TREC' or 'trec' in self.args.task:
            self.num_label = 6
        elif self.args.task in ['yelp','imdb','youtube']:
            self.num_label = 2
        #for i in range(self.num_label):
        self.relation_labels = [x for x in range(self.num_label)]
        self.label2id = {x:x for x in range(self.num_label)}
        self.id2label = {x:x for x in range(self.num_label)}


    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def read_data(self, filename, mode):
        path = filename + '/' + mode + '_data.json'
        path = filename
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def _create_examples(self, data, set_type):
        examples = []
        for i, d in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            text_a = d["text"]
            label = d["label"] if (self.rule == 0 and set_type !='unlabeled') else d["major"]
            #print(text_a, label)
            #if i>10:
            #    assert 0
            if set_type not in ['train', 'unlabeled']:
                label = d["label"]
            if set_type == 'unlabeled':
                label = -1
            true = d["label"]
            if i % 2000 == 0:
                logger.info(d)
            examples.append(InputExample(guid=guid, text_a=text_a, label=label, true = true))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file
        elif mode == 'unlabeled':
            file_to_read = self.args.unlabel_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._create_examples(self.read_data(os.path.join(self.args.data_dir, file_to_read), mode), mode)

class WiCProcessor(object):
    """Processor for the Yelp data set """
    def __init__(self, args):
        self.args = args
        self.rule = self.args.rule
        self.num_label = 2
        self.relation_labels = [x for x in range(self.num_label)]
        self.label2id = {x:x for x in range(self.num_label)}
        self.id2label = {x:x for x in range(self.num_label)}

    def read_data(self, filename, mode):
        path = filename
        with open(path, 'r') as f:
            data = [json.loads(l) for l in f.readlines()]
        return data

    def _create_examples(self, data, set_type):
        examples = []
        for i, d in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            text_a = d["sentence1"]
            text_b = d["sentence2"]
            span_a = (d["start1"], d["end1"])
            span_b = (d["start2"], d["end2"])
            if set_type == 'train' and self.rule == 1:
                label = d["rule_label"]
            elif set_type in ['unlabeled']:
                label = -1
            else:
                label = d['label']
            label = int(label)
            true = int(d["label"])
            if i % 2000 == 0:
                logger.info(d)
            examples.append(WiCInputExample(guid=guid, text_a=text_a, span_a=span_a,
                                         text_b=text_b, span_b=span_b,
                                         label=label, true = true))
            if set_type == 'train':
                examples.append(WiCInputExample(guid=guid, text_a=text_b, span_a=span_b,
                                             text_b=text_a, span_b=span_a,
                                             label=label, true = true))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file
        elif mode == 'unlabeled':
            file_to_read = self.args.unlabel_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._create_examples(self.read_data(os.path.join(self.args.data_dir, file_to_read), mode), mode)
    

class ChemprotProcessor(object):
    """Processor for the Yelp data set """
    def __init__(self, args):
        self.args = args
        self.rule = self.args.rule
        self.num_label = 10
        self.relation_labels = [x for x in range(self.num_label)]
        self.label2id = {x:x for x in range(self.num_label)}
        self.id2label = {x:x for x in range(self.num_label)}

    def read_data(self, filename, mode):
        path = filename
        with open(path, 'r') as f:
            data = [json.loads(l) for l in f.readlines()]
        return data

    def _create_examples(self, data, set_type):
        examples = []
        for i, d in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            text_a = d["text"]
            span_a = (d["start1"], d["end1"])
            span_b = (d["start2"], d["end2"])
            assert d["start1"] >= 0 and d["start2"] >= 0
            if set_type == 'train' and self.rule == 1:
                label = d["major"]
            elif set_type in ['unlabeled']:
                label = -1
            else:
                label = d['label']
            label = int(label)
            label = max(label-1,0)
            true = int(d["label"])
            true = max(true-1,0)
            if i % 2000 == 0:
                logger.info(d)
            examples.append(ReInputExample(guid=guid, text_a=text_a, span_a=span_a,
                                         span_b=span_b, label=label, true = true))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file
        elif mode == 'unlabeled':
            file_to_read = self.args.unlabel_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._create_examples(self.read_data(os.path.join(self.args.data_dir, file_to_read), mode), mode)

processors = {
    "semeval": SemEvalProcessor,
    "yelp2": YelpProcessor,
    "youtube": YoutubeProcessor,
    "imdb": YoutubeProcessor,
    "yelp": YoutubeProcessor,
    "agnews": YoutubeProcessor,
    "agnews1": YoutubeProcessor,
    "TREC": YoutubeProcessor,
    "wic": WiCProcessor,
    "chemprot": ChemprotProcessor,
}

def tokenize_with_span(tokenizer, sent, span):
    _a = tokenizer.tokenize(sent[:span[0]])
    _w = tokenizer.tokenize(sent[span[0]:span[1]])
    _b = tokenizer.tokenize(sent[span[1]:])

    return _a+_w+_b, len(_a),len(_a)+len(_w)

def tokenize_with_2span(tokenizer, sent, span_a, span_b):
    assert span_a[1]<=span_b[0] or span_a[0]>=span_b[1]
    
    if span_a[1]<=span_b[0]:
        _s0 = tokenizer.tokenize(sent[:span_a[0]])
        _wa = tokenizer.tokenize(sent[span_a[0]:span_a[1]])
        _s1 = tokenizer.tokenize(sent[span_a[1]:span_b[0]])
        _wb = tokenizer.tokenize(sent[span_b[0]:span_b[1]])
        _s2 = tokenizer.tokenize(sent[span_b[1]:])
        if not (len(_wa) > 0 and len(_wb) > 0):
            import ipdb; ipdb.set_trace()
        return _s0+_wa+_s1+_wb+_s2, \
                len(_s0),len(_s0)+len(_wa), \
                len(_s0)+len(_wa)+len(_s1),len(_s0)+len(_wa)+len(_s1)+len(_wb)
    else:
        _s0 = tokenizer.tokenize(sent[:span_b[0]])
        _wb = tokenizer.tokenize(sent[span_b[0]:span_b[1]])
        _s1 = tokenizer.tokenize(sent[span_b[1]:span_a[0]])
        _wa = tokenizer.tokenize(sent[span_a[0]:span_a[1]])
        _s2 = tokenizer.tokenize(sent[span_a[1]:])
        if not (len(_wa) > 0 and len(_wb) > 0):
            import ipdb; ipdb.set_trace()
        return _s0+_wb+_s1+_wa+_s2, \
                len(_s0)+len(_wb)+len(_s1),len(_s0)+len(_wb)+len(_s1)+len(_wa), \
                len(_s0),len(_s0)+len(_wb) \

        
    

def convert_examples_to_features_re(examples, max_seq_len, tokenizer,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=0,
                                 sep_token='[SEP]',
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 add_sep_token=False,
                                 mask_padding_with_zero=True,
                                 task = 're'
                                ):
    features = []
    sample_per_example = 3
    for (ex_index, example) in enumerate(examples[:]):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        tokens_a, key_a_start, key_a_end, key_b_start, key_b_end = tokenize_with_2span(tokenizer, example.text_a, example.span_a, example.span_b)
        keys = [0]*len(tokens_a)
        keys[key_a_start:key_a_end] = [1]*(key_a_end-key_a_start)
        keys[key_b_start:key_b_end] = [2]*(key_b_end-key_b_start)

        if add_sep_token:
            tokens_a += [sep_token]
            keys += [0]
        token_type_ids_a = [sequence_a_segment_id] * len(tokens_a)

        tokens = [cls_token] + tokens_a
        keys = [0] + keys
        token_type_ids = [cls_token_segment_id] + token_type_ids_a
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        output_tokens = []
        masked_lm_labels = []


        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        keys = keys + ([0]*padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(keys) == max_seq_len, "Error with input length {} vs {}".format(len(keys), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)


        label_id = int(example.label)
        true = int(example.true)
        e1_mask = [1 if k==1 else 0 for k in keys ]
        e2_mask = [1 if k==2 else 0 for k in keys ]

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            if task == 're':
                logger.info("e1_mask: %s" % " ".join([str(x) for x in e1_mask]))
                logger.info("e2_mask: %s" % " ".join([str(x) for x in e2_mask]))
            #assert 0
        features.append(
            InputFeatures(
                            input_ids=input_ids,
                            e1_mask=e1_mask,
                            e2_mask=e2_mask,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            label_id=label_id,
                            true=true,
                          )
            )


    return features

def convert_examples_to_features_wic(examples, max_seq_len, tokenizer,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=0,
                                 sep_token='[SEP]',
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 add_sep_token=False,
                                 mask_padding_with_zero=True,
                                 task = 're'
                                ):
    features = []
    sample_per_example = 3
    for (ex_index, example) in enumerate(examples[:]):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        tokens_a, key_a_start, key_a_end = tokenize_with_span(tokenizer, example.text_a, example.span_a)
        tokens_b, key_b_start, key_b_end = tokenize_with_span(tokenizer, example.text_b, example.span_b)
        keys_a = [0]*len(tokens_a)
        keys_a[key_a_start:key_a_end] = [1]*(key_a_end-key_a_start)
        keys_b = [0]*len(tokens_b)
        keys_b[key_b_start:key_b_end] = [2]*(key_b_end-key_b_start)

        if add_sep_token:
            tokens_a += [sep_token]
            keys_a += [0]
            tokens_b += [sep_token]
            keys_b += [0]
        token_type_ids_a = [sequence_a_segment_id] * len(tokens_a)
        token_type_ids_b = [sequence_b_segment_id] * len(tokens_b)

        tokens = [cls_token] + tokens_a + tokens_b
        keys = [0] + keys_a + keys_b
        token_type_ids = [cls_token_segment_id] + token_type_ids_a + token_type_ids_b
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        output_tokens = []
        masked_lm_labels = []


        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        keys = keys + ([0]*padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(keys) == max_seq_len, "Error with input length {} vs {}".format(len(keys), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)


        label_id = int(example.label)
        true = int(example.true)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            if task == 're':
                logger.info("e1_mask: %s" % " ".join([str(x) for x in e1_mask]))
                logger.info("e2_mask: %s" % " ".join([str(x) for x in e2_mask]))
            #assert 0

        features.append(
            InputFeatures(
                            input_ids=input_ids,
                            keys=keys,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            label_id=label_id,
                            true=true,
                          )
            )

    return features

def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=0,
                                 sep_token='[SEP]',
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 add_sep_token=False,
                                 mask_padding_with_zero=True,
                                 task = 're'
                                ):
    features = []
    sample_per_example = 3
    for (ex_index, example) in enumerate(examples[:]):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        #print(example.text_a)
        tokens_a = tokenizer.tokenize(example.text_a)
        #print(tokens_a)
        if task == 're':

            e11_p = tokens_a.index("<e1>")  # the start position of entity1
            e12_p = tokens_a.index("</e1>")  # the end position of entity1
            e21_p = tokens_a.index("<e2>")  # the start position of entity2
            e22_p = tokens_a.index("</e2>")  # the end position of entity2

            # Replace the token
            tokens_a[e11_p] = "$"
            tokens_a[e12_p] = "$"
            tokens_a[e21_p] = "#"
            tokens_a[e22_p] = "#"

            # Add 1 because of the [CLS] token
            e11_p += 1
            e12_p += 1
            e21_p += 1
            e22_p += 1

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        if add_sep_token:
            special_tokens_count = 2
        else:
            special_tokens_count = 1
        if len(tokens_a) > max_seq_len - special_tokens_count:
            tokens_a = tokens_a[:(max_seq_len - special_tokens_count)]

        tokens = tokens_a
        if add_sep_token:
            tokens += [sep_token]

        token_type_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        #tokens[0] = "$"
        #tokens[1] = "<e2>"
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        output_tokens = []
        masked_lm_labels = []


        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)


        #assert 0
        if task == 're':
            # e1 mask, e2 mask
            e1_mask = [0] * len(attention_mask)
            e2_mask = [0] * len(attention_mask)

            for i in range(e11_p, e12_p + 1):
                e1_mask[i] = 1
            for i in range(e21_p, e22_p + 1):
                e2_mask[i] = 1

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

        label_id = int(example.label)
        true = int(example.true)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            if task == 're':
                logger.info("e1_mask: %s" % " ".join([str(x) for x in e1_mask]))
                logger.info("e2_mask: %s" % " ".join([str(x) for x in e2_mask]))
            #assert 0

        features.append(
            InputFeatures(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            label_id=label_id,
                            true=true,
                            e1_mask=e1_mask if task == 're' else None,
                            e2_mask=e2_mask if task == 're' else None
                          )
            )

    return features


def load_and_cache_examples(args, tokenizer, mode):
    if 'imdb' in args.task:
        processor = processors["imdb"](args)
    elif 'trec' in args.task:
        processor = processors["TREC"](args)
    else:
        processor = processors[args.task](args)
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}_{}'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len,
            'dist' if args.rule == 1 else 'clean'
        )
    )

    if os.path.exists(cached_features_file) and args.auto_load:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        if args.task_type == 'wic':
            features, = convert_examples_to_features_wic(examples, args.max_seq_len, tokenizer, add_sep_token=args.add_sep_token, task = args.task_type)
        elif args.task_type == 're':
            features = convert_examples_to_features_re(examples, args.max_seq_len, tokenizer, add_sep_token=args.add_sep_token, task = args.task_type)            
        else:
            features = convert_examples_to_features(examples, args.max_seq_len, tokenizer, add_sep_token=args.add_sep_token, task = args.task_type)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_true_ids = torch.tensor([f.true for f in features], dtype=torch.long)
    all_ids = torch.tensor([ _ for _,f in enumerate(features)], dtype=torch.long)
    size = len(features)


    if args.task_type == 're':
        all_e1_mask = torch.tensor([f.e1_mask for f in features], dtype=torch.long)  # add e1 mask
        all_e2_mask = torch.tensor([f.e2_mask for f in features], dtype=torch.long)  # add e2 mask
        dataset = TensorDataset(all_input_ids, all_attention_mask,
                                all_token_type_ids, all_label_ids, all_e1_mask, all_e2_mask)
    elif args.task_type == 'wic':
        all_keys = torch.tensor([f.keys for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_mask,
                                all_token_type_ids, all_label_ids, all_ids, all_true_ids, all_keys)
    else:
        dataset = TensorDataset(all_input_ids, all_attention_mask,
                                all_token_type_ids, all_label_ids, all_ids, all_true_ids)
    return dataset, processor.relation_labels, processor.num_label, processor.id2label, processor.label2id, size


def load_and_cache_unlabeled_examples(args, tokenizer, mode, train_size = 100):
    if 'imdb' in args.task:
        processor = processors["imdb"](args)
    elif 'trec' in args.task:
        processor = processors["TREC"](args)
    else:
        processor = processors[args.task](args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}_unlabel_{}'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len,
            'dist' if args.rule == 1 else 'clean'
        )
    )

    if os.path.exists(cached_features_file) and args.auto_load:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)

        assert mode == "unlabeled"
        examples = processor.get_examples("unlabeled")
        if args.task_type == 'wic':
            features = convert_examples_to_features_wic(examples, args.max_seq_len, tokenizer, add_sep_token=args.add_sep_token, task = args.task_type)
        elif args.task_type == 're':
            features = convert_examples_to_features_re(examples, args.max_seq_len, tokenizer, add_sep_token=args.add_sep_token, task = args.task_type)
        else:
            features = convert_examples_to_features(examples, args.max_seq_len, tokenizer, add_sep_token=args.add_sep_token, task = args.task_type)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_true_ids = torch.tensor([f.true for f in features], dtype=torch.long)
    all_ids = torch.tensor([_+train_size for _ ,f in enumerate(features)], dtype=torch.long)


    if args.task_type == 're':
        all_e1_mask = torch.tensor([f.e1_mask for f in features], dtype=torch.long)  # add e1 mask
        all_e2_mask = torch.tensor([f.e2_mask for f in features], dtype=torch.long)  # add e2 mask
        dataset = TensorDataset(all_input_ids, all_attention_mask,
                                all_token_type_ids, all_label_ids, all_e1_mask, all_e2_mask)
    elif args.task_type == 'wic':
        all_keys = torch.tensor([f.keys for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_mask,
                                all_token_type_ids, all_label_ids, all_ids, all_true_ids, all_keys)
    else:
        dataset = TensorDataset(all_input_ids, all_attention_mask,
                                all_token_type_ids, all_label_ids, all_ids, all_true_ids)

    return dataset, len(features)
