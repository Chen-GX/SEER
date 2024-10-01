
export CUDA_VISIBLE_DEVICES="1"
EXEC="../code/Controller.py"

task="task_2"
exp_dir='../result/run'
epochs=20
seed=42


python $EXEC \
--task $task \
--epochs $epochs \
--seed $seed \
--exp_dir $exp_dir \
--adafactor \
--save_model