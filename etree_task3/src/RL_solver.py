import torch
import torch.nn.functional as F
import copy
import numpy as np
import wandb
from RL_state import Action
from policy_net import LMActorCriticPolicy
from normalization import Normalization
from reward import score_aligned_entail_tree_proof_onlyIR
from memorybuffer import MemoryBuffer
import logging
logger = logging.getLogger(__name__)

class Solver(object):
    def __init__(self, args) -> None:
        self.args = args
        self.policy = LMActorCriticPolicy(args)
        self.policy.to(args.device)
        if args.disable_dropout:
            self.disable_dropout()
            
        self.create_optimizer()
        # policy params
        self.load_policy_params()

        # init buffer
        self.buffer = MemoryBuffer(args)

        if args.use_reward_norm:  # Trick 3:reward normalization
            self.reward_norm = Normalization(shape=1)
        elif args.use_reward_scaling:  # Trick 4:reward scaling
            self.max_reward = self.args.max_reward

        assert sum([args.use_reward_norm, args.use_reward_scaling]) <= 1, "More than one variable is True!"

    def disable_dropout(self):
        for module in self.policy.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0

    def load_policy_params(self):
        # load policy generate params
        self.linearize_state_form = self.args.linearize_state_form
        self.generate_args = {
            'num_beams': self.args.controller_num_return_sequences,
            'num_return_sequences': self.args.controller_num_return_sequences,
        }
        

    def create_optimizer(self):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        names = list(self.policy.policy_model.named_parameters())
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in names if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in names if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.actor_optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=self.args.actor_lr, eps=self.args.adamw_eps)

        names = list(self.policy.value_model.named_parameters()) + list(self.policy.value_head.named_parameters())
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in names if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in names if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.critic_optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=self.args.critic_lr, eps=self.args.adamw_eps)

    def get_action(self, state):
        # 获得state的输入
        linear_state = [state.linearize_state(form=self.linearize_state_form)]
        step_action = self.policy.seq2seq_generate(linear_state, generate_args=self.generate_args, state=state)
        return step_action
        
    def reason_one_step(self, etree, state, cur_step):
        # self.model.eval()
        with torch.no_grad():
            # state: 当前步骤
            step_action = self.get_action(state)
            if step_action['type'] == Action.end:  # 是不会添加新的state的，所以[:-1]会直接去掉这个动作
                if self.args.verbose:
                    logger.info(f'step {cur_step}: {step_action["type"]}')
                done = True
            elif step_action['type'] == Action.reason:
                next_state, done = etree.get_next_state(state, step_action['step']['pre_id'])  
                if self.args.verbose:
                    logger.info(f'step {cur_step}: {step_action["type"]} {" & ".join(step_action["step"]["pre_id"])} -> {state.intermediate_conclusion} || {state.intermediate_bleurt_score:.3f}')
                etree.state_list.append(next_state)
            else:
                done = True
        return done

    def reason_etree(self, etree):
        # 先前向推理，得到etree的reward和state
        self.policy.eval()
        cur_step = 0
        if self.args.verbose:
            logger.info(f"{etree.data_item['id']}")
        while cur_step < 20:
            done = self.reason_one_step(etree, etree.state_list[cur_step], cur_step)
            cur_step += 1
            if done:
                break

        # 构建整棵树 pred_tree
        etree.build_proof_task3()

        if len(etree.pred) > 0:
            # 和gold tree之间计算reward
            reward_list, key_state_list = score_aligned_entail_tree_proof_onlyIR(etree, etree.pred, [etree.data_item['task1']['proof']], 'proof', etree.data_item['task1'], etree.prediction_json)
            
            if self.args.use_reward_norm:
                acc_reward_list = [self.reward_norm(r) for r in acc_reward_list]
            elif self.args.use_reward_scaling:
                acc_reward_list = [r / self.max_reward for r in acc_reward_list]

            return reward_list, key_state_list, cur_step
        else:
            return None, None, None
        


    def to_tensor(self, dones, actions, log_probs, rewards):
        dones_tensor = torch.tensor(dones, dtype=torch.float, device=self.args.device).view(-1, 1)
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.args.device).view(-1, 1)
        log_probs_tensor = torch.tensor(log_probs, device=self.args.device).view(-1, 1)
        rewards_tensor = torch.tensor(rewards, device=self.args.device).view(-1, 1)
        return dones_tensor, actions_tensor, log_probs_tensor, rewards_tensor

    def update_model(self):
        self.policy.train()
        actor_loss_lst, critic_loss_lst = [], []
        for _ in range(self.buffer.k_epochs):
            for s, s_, dones, actions, candidate_action_texts, log_probs, rewards in self.buffer.sample_sequentially():
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                dones, actions, old_log_probs, rewards = self.to_tensor(dones, actions, log_probs, rewards)
                new_log_probs, new_entropys, values, values_ = self.policy.evaluation_actions(s=s, s_=s_, candidate_action_texts=candidate_action_texts, actions=actions)
                # 转换为tensor
                td_target = rewards + self.args.gamma * values_ * (1 - dones)  # R   也是V网络的目标
                td_delta = (td_target - values).detach()
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * td_delta
                surr2 = torch.clip(ratio, 1 - self.args.eps, 1 + self.args.eps) * td_delta
                actor_loss = torch.mean(-torch.min(surr1, surr2) - self.args.entropy_coef * new_entropys)
                critic_loss = torch.mean(F.mse_loss(values, td_target.detach()))

                # 梯度会累积直到参数更新
                actor_loss.backward()
                critic_loss.backward()
                
                if self.args.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.policy.policy_model.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(self.policy.value_model.parameters(), 0.5)

                self.actor_optimizer.step()
                self.critic_optimizer.step()
                # 计算每一步的损失，累加到总损失中
                actor_loss_lst.append(actor_loss.item())
                critic_loss_lst.append(critic_loss.item())

                if self.args.wandb:
                    wandb.log({"Train/actor_loss": actor_loss.item(), "Train/critic_loss": critic_loss.item()})
        
        return np.mean(actor_loss_lst), np.mean(critic_loss_lst)

    def inference_whole_tree(self, etree):
        self.policy.eval()
        cur_step = 0
        if self.args.verbose:
            logger.info(f"{etree.data_item['id']}")

        while cur_step < 20:  # 因为FAME有可能会合并多个，所以不能再按照步长来解决
            done = self.reason_one_step(etree, etree.state_list[cur_step], cur_step)
            cur_step += 1
            if done:
                break
        
        # 构建整棵树 pred_tree
        etree.build_proof_task3()





