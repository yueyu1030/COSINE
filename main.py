import argparse
import os
from trainer import Trainer
from utils import init_logger, load_tokenizer, MODEL_CLASSES, MODEL_PATH_MAP
from data_loader_new import load_and_cache_examples, load_and_cache_unlabeled_examples


def main(args):
    init_logger()
    tokenizer = load_tokenizer(args)


    train_dataset, relation_labels, num_labels, id2label, label2id, train_size  = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset, relation_labels, num_labels, id2label, label2id, dev_size = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset, relation_labels, num_labels, id2label, label2id,test_size = load_and_cache_examples(args, tokenizer, mode="test")
    unlabeled_dataset, unlabeled_size = load_and_cache_unlabeled_examples(args, tokenizer, mode = 'unlabeled', train_size = train_size)
    print(relation_labels, 'number of labels:', num_labels)
    print('train_size:', train_size)
    print('dev_size:', dev_size)
    print('test_size:', test_size)
    print('unlabel_size:', unlabeled_size)
    import time
    time.sleep(1.6)

    #assert 0
    trainer = Trainer(args, train_dataset=train_dataset, dev_dataset=dev_dataset,test_dataset=test_dataset, labelset = relation_labels, \
            unlabeled = unlabeled_dataset, \
            #masked_train_dataset = masked_train_dataset, masked_dev_dataset = masked_dev_dataset, \
            #masked_test_dataset = masked_test_dataset, masked_unlabeled_dataset = masked_unlabeled_dataset, \
            num_labels = num_labels, id2label = id2label, label2id = label2id, data_size = train_size
            )

    if args.do_train:
        if args.pretrain:
            trainer.pretrain()
        if args.method in ['clean', 'noisy', "noise"]:
            trainer.train()
            trainer.save_features()
        elif args.method == 'selftrain':
            trainer.train()
            trainer.selftrain(soft = args.soft_label, adv = args.add_adv)
        
    if args.do_eval:
        trainer.evaluate('test')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default='clean', type=str, help="which method to use")
    parser.add_argument("--task_type", default="re", type=str, help="the specific task in IE: [ tc | re | ner ]")
    parser.add_argument("--gpu", default='0,1,2,3', type=str, help="which gpu to use")
    parser.add_argument("--task", default="semeval", type=str, help="The name of the task to train")
    parser.add_argument("--rule", default=1, type=int, help="Use rule or not")
    parser.add_argument("--data_dir", default="./data", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to model")
    parser.add_argument("--eval_dir", default="./eval", type=str, help="Evaluation script, result directory")
    parser.add_argument("--train_file", default="train.tsv", type=str, help="Train file")
    parser.add_argument("--dev_file", default="dev.tsv", type=str, help="dev file")
    parser.add_argument("--test_file", default="test.tsv", type=str, help="Test file")
    parser.add_argument("--unlabel_file", default="unlabeled.tsv", type=str, help="Test file")
    parser.add_argument("--label_file", default="label.txt", type=str, help="Label file")

    parser.add_argument("--model_type", default="roberta", type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--auto_load", default=1, type=int, help="Auto loading the model or not")

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--max_seq_len", default=512, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="Training steps for initialization.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument('--logging_steps', type=int, default=20, help="Log every X updates steps.")
    parser.add_argument('--self_train_logging_steps', type=int, default=20, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--add_sep_token", action="store_true", help="Add [SEP] token at the end of the sentence")

    # self-training
    parser.add_argument('--soft_label', type = int, default = 1, help = 'whether soft label (0 for hard, 1 for soft)')
    parser.add_argument("--soft_label_weight", default=1.0, type=float, help="iters for pretrains")
    parser.add_argument('--self_training_eps', type = float, default = 0.8, help = 'threshold for confidence')
    parser.add_argument('--self_training_power', type = float, default = 2, help = 'power of pred score')
    parser.add_argument('--self_training_reg', type = float, default = 0, help = 'confidence smooth power')
    parser.add_argument('--self_training_contrastive_weight', type = float, default = 0, help = 'contrastive learning weight')

    parser.add_argument('--self_training_max_step', type = int, default = 10000, help = 'the maximum step (usually after the first epoch) for self training')    
    parser.add_argument('--distmetric', type = str, default = "l2", help = 'distance type. Choices = [cos, l2]')
    parser.add_argument('--self_training_label_mode', type = str, default = "hard", help = 'pseudo label type. Choices = [hard, soft]')
    parser.add_argument('--self_training_update_period', type = int, default = 100, help = 'update period')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    if args.method == 'clean':
        args.rule = 0
    else:
        args.rule = 1 

    main(args)
