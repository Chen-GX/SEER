import os, sys
import os.path as osp
os.chdir(sys.path[0])
import logging
import torch
import time 
import wandb
import copy
import numpy as np
import random
import json
from tqdm import tqdm
from arguments import get_args
from utils import set_seed, load_data
from RL_etree import eTree
from RL_solver import Solver
from RL_agent import Agent
from RL_state import Action
sys.path.append("..")
from entailment_bank_get_metric.eval.run_scorer import get_etree_metric

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        # for gpu in gpus:
        #   tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

logger = logging.getLogger(__name__)

def main(args):
    set_seed(args.seed)
    # wandb
    if args.wandb:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_project_name,
            # track hyperparameters and run metadata
            config=args,
            name=args.wandb_name,
            dir=args.wandb_dir,
        )

    if torch.cuda.device_count() == 1:
        args.device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else "cpu")
        args.no_grad_device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else "cpu")      
    elif torch.cuda.device_count() == 2:
        args.device = torch.device('cuda:1' if torch.cuda.is_available() and args.gpu else "cpu")
        args.no_grad_device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else "cpu")
        logger.info(f'two gpu')
    else:
        raise NotImplementedError
    
    # load solver
    solver = Solver(args)
    agent = Agent(args)

    # load data
    train, dev, test = load_data(args)  # 不进行检索

    # 记录指标数据
    best_step = -1
    best_overall_acc = -1
    best_metric = None
    best_test_step = -1
    best_test_overall_acc = -1
    best_test_metric = None
    update_counter = 0
    for epoch in range(args.epochs):
        begin_time = time.time()
        train_epoch_etree, actor_loss_lst, critic_loss_lst = [], [], []
        random.shuffle(train)  # 每个epoch都打乱etree
        for data_item in tqdm(train):
            begin_item_time = time.time()
            etree = eTree(args, copy.deepcopy(data_item), agent)
            # 采样
            reward_list, state_list, step_cnt = solver.reason_etree(etree)
            train_epoch_etree.append(etree)  # 加入list为了后续记录metric
            if reward_list is None:
                logger.info(f'{etree.data_item["id"]} | None ')
                continue
            
            for reward, state in zip(reward_list, state_list):
                solver.buffer.add(reward, state)
                if solver.buffer.is_full():
                    logger.info(f'buffer full')
                    actor_loss, critic_loss = solver.update_model()
                    actor_loss_lst.append(actor_loss)
                    critic_loss_lst.append(critic_loss)
                    solver.buffer.reset()  # 清空buffer
                    update_counter += 1  # 更新计数器
                    if update_counter % args.evaluate_steps == 0:
                        logger.info(f'start evaluate {update_counter}')
                        if args.do_dev:
                            dev_epoch_metric = validate_model(solver, agent, dev, update_counter, 'dev')
                            if dev_epoch_metric['proof-overall']['acc'] > best_overall_acc:
                                best_step = update_counter
                                best_overall_acc = dev_epoch_metric['proof-overall']['acc']
                                best_metric = copy.deepcopy(dev_epoch_metric)
                                logger.info(f"best dev step {update_counter} with {dev_epoch_metric}")
                                solver.policy.save_model(osp.join(args.output_dir, 'best_model'))
                            else:
                                logger.info(f"dev step {update_counter} with {dev_epoch_metric}")

                        if args.do_test:
                            test_epoch_metric = validate_model(solver, agent, test, update_counter, 'test')
                            if test_epoch_metric['proof-overall']['acc'] > best_test_overall_acc:
                                best_test_step = update_counter
                                best_test_overall_acc = test_epoch_metric['proof-overall']['acc']
                                best_test_metric = copy.deepcopy(test_epoch_metric)
                                logger.info(f"best test step {update_counter} with {test_epoch_metric}")
                                solver.policy.save_model(osp.join(args.output_dir, 'best_test', 'best_model'))
                            else:
                                logger.info(f"test step {update_counter} with {test_epoch_metric}")
                        
                        # 退火entropy_coef
                        if  args.entropy_do_decline and update_counter % args.entropy_step == 0:
                            args.entropy_coef /= args.entropy_decline_factor
                            args.entropy_coef = max(args.entropy_coef, 1e-8)  # 保证不会低于某个阈值
                            logger.info(f'decline entropy_coef: {args.entropy_coef}')
                        

            # 记录一个样本的结果，也便于监控哪个id报错了
            logger.info(f'[{update_counter}/{args.evaluate_steps}] {etree.data_item["id"]} | key steps {len(reward_list)} | T: {time.time() - begin_item_time:.2f}')

        # # 一个epoch结束，存储结果
        train_epoch_metric = save_and_metric_task1_2(agent, train_epoch_etree, epoch, 'train')
        
        # 记录一个epoch的结果
        if args.wandb:
            wandb.log({"Train/epoch_actor_loss": np.mean(actor_loss_lst), "Train/epoch_critic_loss": np.mean(critic_loss_lst), "Train/Overall_acc": train_epoch_metric['proof-overall']['acc']})
        
        logger.info(f"epoch {epoch} train Overall_acc: {train_epoch_metric['proof-overall']['acc']:.4f} | epoch_policy_loss {np.mean(actor_loss_lst):.6f} | epoch_critic_loss {np.mean(critic_loss_lst):.6f} | time cost: {(time.time() - begin_time) / 60 :.1f} minutes")
        
        if epoch % 20 == 0 and epoch != 0:
            # force save buffer
            agent.final_write_buffer()

    agent.final_write_buffer()

    logger.info(f'best_dev_step: {best_step}\n{best_metric}')
    logger.info(f'best_test_step: {best_test_step}\n{best_test_metric}')

def validate_model(solver, agent, dataset, steps, data_split):
    # 在验证集上验证
    begin_time = time.time()
    epoch_tree = []
    for item in tqdm(dataset):
        etree = eTree(args, copy.deepcopy(item), agent)
        begin_episode_time = time.time()
        solver.inference_whole_tree(etree)
        logger.info(f'{etree.data_item["id"]} | T {time.time() - begin_episode_time:.1f}')
        epoch_tree.append(etree)
    
    epoch_metric = save_and_metric_task1_2(agent, epoch_tree, steps, data_split)
    if args.wandb:
        wandb.log({f"{data_split}/overall_acc": epoch_metric['proof-overall']['acc']})
    logger.info(f'steps {steps} {data_split} time cost: {(time.time() - begin_time) / 60:.1f} minutes')
    return epoch_metric


def save_and_metric_task1_2(agent, etree_list, steps, data_split):
    # 一个epoch结束，存储结果
    with open(osp.join(args.output_dir, 'epoch_tree', f'{data_split}', f'prediction_{steps}.jsonl'), 'w') as f:
        for etree in etree_list:
            text = {
                "id": etree.data_item['id'],
                "angle": [["question", "answer", "hypothesis", "context"], ["proof"]],
                "prediction": "$proof$ = " + etree.pred
            }
            json.dump(text, f)
            f.write('\n')

    epoch_metric = get_etree_metric(
        task=args.task,
        output_dir=osp.join(args.output_dir, 'epoch_tree', f'{data_split}', str(steps)),
        split=data_split,
        prediction_file=osp.join(args.output_dir, 'epoch_tree', f'{data_split}', f'prediction_{steps}.jsonl'),
        bleurt_checkpoint=None,
        use_bleurt_buffer=True,
        bleurt_buffer_file="../entailment_bank_get_metric/data/buffer/buffer_bleurt_all.json",
        bleurt_scorer=agent.bleurt_scorer,
    )
    return epoch_metric

if __name__=="__main__":
    args = get_args()
    main(args)
