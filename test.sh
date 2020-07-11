export CUDA_VISIBLE_DEVICES=6,7
python run_qa.py \
--model_name_or_path /home/none404/hm/Model/Roberta_zh_l12 \
--do_test \
--k 10 \
--data_dir ../data/data_0 \
--test_dir ../data/test1.csv \
--passage_dir ../data/passage1.csv \
--output_dir ./output \
--max_seq_length 512 \
--max_question_length 96 \
--eval_steps 50 \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--train_steps 1000