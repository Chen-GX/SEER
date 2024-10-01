
export CUDA_VISIBLE_DEVICES="4"
EXEC="../code/Controller.py"

task="task_1"
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