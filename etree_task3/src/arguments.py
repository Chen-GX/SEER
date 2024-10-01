import os
import os.path as osp
import time
import argparse
from log_utils import log_params

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False


def get_args():
    parser = argparse.ArgumentParser()  # 参数解释器

    ## 预训练模型参数
    parser.add_argument('--cache_dir', type=str, default='../../exp/ptm')
    # ../../exp/Controller/Iter5/5Vwv6zll   ../../exp/Controller/Iter0/pGhkkrVa
    parser.add_argument('--policy_path', type=str, default="../../exp/Controller/Iter0/pGhkkrVa")
    parser.add_argument('--linearize_state_form', type=str, default="QAHPS")
    parser.add_argument("--controller_num_return_sequences", type=int, default=5) 
    
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--k_epochs', type=int, default=2) 
    parser.add_argument('--buffer_size', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--evaluate_steps', type=int, default=1)
    parser.add_argument('--debug_max_num', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=0.99)  # 判断是否得到H的阈值

    # 3. Entailment Module
    parser.add_argument('--entailment_module_exp_dir', type=str, default="../../exp/EntailmentModule/IK3tEKPo/")
    parser.add_argument('--buffer_file', type=str, default="../buffer/buffer_dict_entail_etree_all.json")
    parser.add_argument('--use_buffer', type=str2bool, default=True)
    parser.add_argument('--write_buffer_step', type=int, default=20000)
    # 4. bleurt Module
    parser.add_argument('--bleurt_path', type=str, default='../../exp/bleurt-large-512')
    parser.add_argument('--use_bleurt_buffer', type=str2bool, default=True)
    parser.add_argument('--bleurt_buffer_file', type=str, default='../buffer/buffer_bleurt_all.json')

    # retrieve Module  
    parser.add_argument("--corpus_path", type=str, default="../../exp/corpus/supporting_data/preprocessed_new_corpus.json")
    parser.add_argument("--retriever_path_or_name", type=str, default="../../exp/retriever/v1")  

    # RL参数
    parser.add_argument('--gamma', type=float, default=0.95)  # 折扣系数
    parser.add_argument('--actor_lr', type=float, default=2e-6)
    parser.add_argument('--critic_lr', type=float, default=2e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--critic_active', type=str2bool, default=True)
    parser.add_argument('--check_premise_overlap', type=str2bool, default=False)  # 是否进行premise的overlap检查
    parser.add_argument('--eps', type=float, default=0.2)  # PPO clip的参数
    parser.add_argument('--disable_dropout', type=str2bool, default=True)
    parser.add_argument('--strategy', type=str, default='sample', help='greedy sample')
    parser.add_argument("--entropy_coef", type=float, default=1e-4)
    parser.add_argument("--entropy_step", type=int, default=1100)  # 更新多少次后降低熵探索
    parser.add_argument("--entropy_decline_factor", type=int, default=100)  # 每次衰退多少倍
    parser.add_argument("--entropy_do_decline", type=str2bool, default=False)  # 是否进行衰退
    parser.add_argument('--use_grad_clip', type=str2bool, default=False)
    parser.add_argument('--adamw_eps', type=float, default=1e-8)
    parser.add_argument('--reward_value', type=float, default=1)
    parser.add_argument('--penalize_value', type=float, default=-1)
    parser.add_argument('--redundant_value', type=float, default=-0.5)
    parser.add_argument("--v_strategy", type=str, default="max") # V值网络聚合信息  mean max last token

    # RL trick
    parser.add_argument("--use_reward_norm", type=str2bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=str2bool, default=False, help="Trick 4:reward scaling (/ max reward)")
    parser.add_argument("--max_reward", type=float, default=5.0, help="Trick 4:reward scaling")

    # datasets 参数
    parser.add_argument('--task', type=str, default='task_3')
    parser.add_argument('--data_dir', type=str, default='../../data/entailment_trees_emnlp2021_data_v3/dataset')

    # 存储参数
    parser.add_argument("--output_dir", type=str, default="../../output_dir/etree_task3/test")

    # 其他参数
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=str2bool, default=True)
    parser.add_argument('--do_train', type=str2bool, default=True)
    parser.add_argument('--do_dev', type=str2bool, default=False)
    parser.add_argument('--do_test', type=str2bool, default=True)

    # wandb 参数
    parser.add_argument('--wandb', action='store_true', help='Whether to use wandb')
    parser.add_argument('--wandb_project_name', type=str, default='etree_task3')
    parser.add_argument('--wandb_name', type=str, default='')
    parser.add_argument('--wandb_dir', type=str, default='../../output_dir/etree_task3/logs')

    parser.add_argument('--verbose', type=str2bool, default=True)

    args = parser.parse_args()  # 解析参数

    # 时间戳后缀，
    timestamp = time.strftime("%m-%d_%H-%M-%S",time.localtime())
    args.timestamp = timestamp
    policy_name = args.policy_path.split('/')[-1]

    args.wandb_name = f"{policy_name}_lr_{args.actor_lr}_{args.critic_lr}_ep_{args.epochs}_{args.timestamp}"

    args.output_dir = osp.join(args.output_dir, args.task, policy_name, args.timestamp)  # , args.timestamp
    # 创建存储文件目录
    if args.do_train:
        os.makedirs(osp.join(args.output_dir, 'epoch_tree', 'train'), exist_ok=True)
    if args.do_dev:
        os.makedirs(osp.join(args.output_dir, 'epoch_tree', 'dev'), exist_ok=True)
    if args.do_test:
        os.makedirs(osp.join(args.output_dir, 'epoch_tree', 'test'), exist_ok=True)
    os.makedirs(osp.join(args.output_dir, 'best_model'), exist_ok=True)
    os.makedirs(osp.join(args.output_dir, 'best_test', 'best_model'), exist_ok=True)

    os.system(f"cp -r ../src {args.output_dir}")

    os.makedirs(args.wandb_dir, exist_ok=True)
    log_params(args)
    return args