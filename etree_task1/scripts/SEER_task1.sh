#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"
EXEC="../src/train.py"

debug_max_num=0
epochs=100
k_epochs=2
buffer_size=12
batch_size=3
evaluate_steps=100
policy_path='../../exp/Controller/task1/76fh6rEr'  # the policy after warm-up
output_dir="../../output_dir/etree_task1/run"


actor_lr=2e-6
critic_lr=2e-6

entropy_coef=1e-4
entropy_step=1100
entropy_decline_factor=100
entropy_do_decline=False

adamw_eps=1e-8
reward_value=1
penalize_value=-1
redundant_value=-0.5
threshold=0.99
v_strategy="max"

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

