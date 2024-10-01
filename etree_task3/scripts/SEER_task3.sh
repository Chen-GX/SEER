#!/bin/bash
# MY_PYTHON="/home/coder/miniconda/envs/etree/bin/python"
# conda activate etree
# which python

export CUDA_VISIBLE_DEVICES="0"
EXEC="../src/train.py"

debug_max_num=0
epochs=100
k_epochs=2
buffer_size=12
batch_size=3
evaluate_steps=100
policy_path='../../exp/Controller/Iter0/pGhkkrVa'  # ../../exp/Controller/Iter5/5Vwv6zll
output_dir="../../output_dir/PPO_etree_task3_buffer/run"
corpus_path="../../exp/corpus/supporting_data/preprocessed_new_corpus.json"

# 学习参数
actor_lr=2e-6
critic_lr=2e-6
# 熵探索的参数
entropy_coef=1e-4
entropy_step=1100
entropy_decline_factor=100
entropy_do_decline=False
# 
adamw_eps=1e-8
reward_value=1
penalize_value=-1
redundant_value=-0.5
threshold=0.99
v_strategy="mean"

python $EXEC \
--debug_max_num $debug_max_num \
--epochs $epochs \
--k_epochs $k_epochs \
--buffer_size $buffer_size \
--batch_size $batch_size \
--evaluate_steps $evaluate_steps \
--output_dir $output_dir \
--policy_path $policy_path \
--actor_lr $actor_lr \
--critic_lr $critic_lr \
--corpus_path $corpus_path \
--entropy_coef $entropy_coef \
--entropy_step $entropy_step \
--entropy_decline_factor $entropy_decline_factor \
--entropy_do_decline $entropy_do_decline \
--adamw_eps $adamw_eps \
--reward_value $reward_value \
--penalize_value $penalize_value \
--redundant_value $redundant_value \
--threshold $threshold \
--v_strategy $v_strategy \
--wandb

