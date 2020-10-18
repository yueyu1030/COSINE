task=agnews
gpu=0
method=selftrain
max_seq_len=128
batch_size=32
echo ${method}
python3 main.py \
	--do_train \
	--do_eval \
	--task=${task} \
	--train_file=train_data.json \
	--dev_file=dev_data.json \
	--test_file=test_data.json \
	--unlabel_file=unlabeled_data.json \
	--task_type=tc \
	--data_dir="data/${task}" \
	--rule=1 \
	--logging_steps=100 \
	--self_train_logging_steps=100 \
	--gpu="${gpu}" \
	--num_train_epochs=3 \
	--weight_decay=1e-4 \
	--method=${method} \
	--batch_size=${batch_size} \
	--max_seq_len=${max_seq_len} \
	--auto_load=1 \
	--bond_eps=0.6 \
	--self_training_update_period=250 \
	--max_steps=150 \
	--self_training_max_step=2500 \
	--self_training_power=2 \
	--self_training_confreg=0.1 \
	--self_training_contrastive_weight=1 \
	--distmetric='cos' \