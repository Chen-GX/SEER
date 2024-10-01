import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os.path as osp
import argparse
import glob
import os
from torch.distributions.categorical import Categorical
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5EncoderModel
from RL_state import Action
import logging
logger = logging.getLogger(__name__)


def check_nan_and_inf(tensor):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        return True
    else:
        return False

class LMActorCriticPolicy(nn.Module):
    def __init__(self, args):
        super(LMActorCriticPolicy, self).__init__()
        self.args = args
        self.policy_model, self.value_model, self.tokenizer, self.model_args = self.load_controller()
        if args.critic_active:
            self.value_head = nn.Sequential(
                nn.Linear(self.value_model.config.hidden_size, self.value_model.config.hidden_size // 2),
                nn.Tanh(),
                nn.Linear(self.value_model.config.hidden_size // 2, 1),
            )
        else:
            self.value_head = nn.Linear(self.value_model.config.hidden_size, 1)

    def load_controller(self):
        # read config
        config = json.load(open(osp.join(self.args.policy_path,'config.json')))
        model_config = json.load(open(osp.join(self.args.policy_path,'model.config.json')))
        model_args = argparse.Namespace(**config)
        parm_files = glob.glob(f"{self.args.policy_path}/*.pth")
        assert len(parm_files) == 1
        # load policy model
        logger.info(f"Loading model from {parm_files[0]}")
        if model_args.model_name_or_path in ['t5-large','t5-base','t5-small']:
            model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path, cache_dir=self.args.cache_dir, local_files_only=True)
            value_model = T5EncoderModel.from_pretrained(model_args.model_name_or_path, cache_dir=self.args.cache_dir, local_files_only=True)
            tokenizer = T5Tokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=self.args.cache_dir, local_files_only=True)
        else:
            raise NotImplementedError

        model.config.update(model_config)
        value_model.config.update(model_config)

        # load trained parameters
        state_dict = torch.load(parm_files[0])
        model.load_state_dict(state_dict)
        value_model.load_state_dict(state_dict, strict=False)
        return model, value_model, tokenizer, model_args

    def load_best_model(self, best_path):
        # load trained parameters
        policy_model_dict = torch.load(osp.join(best_path, 'policy_best_model.pth'))
        self.policy_model.load_state_dict(policy_model_dict)
        value_model_dict = torch.load(osp.join(best_path, 'value_best_model.pth'))
        self.value_model.load_state_dict(value_model_dict, strict=False)
        self.value_head.load_state_dict(value_model_dict, strict=False)

    def save_model(self, path):
        '''Save the trained model.
        
        Args:
            model: A PyTorch model instance which is to be saved.
            path (str): The path where the model should be saved.

        Returns:
            None
        '''
        # 保存 policy_model 的参数
        # os.makedirs(path, exist_ok=True)
        torch.save(self.policy_model.state_dict(), osp.join(path, 'policy_best_model.pth'))

        # 为了将 value_model 和 value_head 的参数保存在同一个文件中，
        # 我们创建一个新的 state_dict 并将两者的参数都添加进去
        combined_state_dict = self.value_model.state_dict()
        combined_state_dict.update(self.value_head.state_dict())
        torch.save(combined_state_dict, osp.join(path, 'value_best_model.pth'))
        # torch.save(self.policy_model.state_dict(), path)


    def get_legal_action(self, action_strs, state):
        result = [ac for ac in action_strs if state.check_action_executable(ac)]
        return list(dict.fromkeys(result))  # 最多返回5个action  去重
    
    def seq2seq_generate(self, input_sents, generate_args={}, state=None):

        model, tokenizer = self.policy_model, self.tokenizer
        model.eval()
        generate_args['max_length'] = 128
        generate_args['num_return_sequences'] = generate_args.get('num_return_sequences', 1)
        generate_args['return_dict_in_generate'] = True

        input_batch = tokenizer(input_sents, add_special_tokens=True, return_tensors='pt', padding='longest', max_length=512, truncation=True,)
        input_batch = input_batch.to(model.device)

        # generate
        outputs = model.generate(input_ids = input_batch['input_ids'], attention_mask = input_batch['attention_mask'],  **generate_args)
        action_strs = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        legal_action_text = self.get_legal_action(action_strs, state)

        if len(legal_action_text) == 0:
            # 强制生成合法动作
            legal_action_text = state.generate_legal_action()
        if 'end' not in legal_action_text:
            legal_action_text.append("end")

        # legal_action_text = ['end']

        # 计算log_probs 并且采样, 得到action, log_prob, entropy
        action, log_prob = self.forward(input_text=input_sents, target_text=legal_action_text)

        # 将状态更新到state上
        state.s = input_sents[0]
        state.action_text = legal_action_text[action]
        state.action = action
        state.candidate_action_text = legal_action_text
        state.log_prob = log_prob

        step_action = Action.parse_action(legal_action_text[action])

        return step_action

    def forward(self, input_text: list, target_text: list):
        # 将input_text转换为t5模型的输入格式
        encodings = self.tokenizer(input_text * len(target_text), max_length=512, return_tensors="pt", padding="longest", truncation=True).to(self.args.device)
        encodings = {k: v for k, v in encodings.items()}
        # 将target_text转换为t5模型的输出格式
        labels = self.tokenizer(target_text, return_tensors="pt", max_length=512, padding="longest", truncation=True)['input_ids']
        # 由labels生成decoder_input_ids，需要在前面补0使得长度与labels相同
        decoder_input_ids = torch.cat([torch.zeros_like(labels[:, :1]), labels[:, :-1]], dim=-1).to(self.args.device)
        action, log_prob = self.forward_policy(encodings, labels, decoder_input_ids)

        return action, log_prob
        

    def forward_policy(self, encodings, labels, decoder_input_ids):
        # 计算生成text的概率
        outputs = self.policy_model(**encodings, decoder_input_ids=decoder_input_ids)
        # 使用logits计算生成labels的概率，logits的shape为[batch_size,seq_len,vocab_size]
        logits = outputs["logits"]

        # 计算log prob
        log_logits = F.log_softmax(logits, dim=-1)  # torch.Size([6, 7, 32128])

        seq_len = torch.arange(labels.shape[-1]).expand(labels.shape[0], -1).to(self.args.device)
        log_labels_token_prob_list = log_logits[torch.arange(labels.shape[0]).unsqueeze(1), seq_len, labels]

        # 将 padding 位置的 log prob 设置为 0
        log_labels_token_prob_list[labels == 0] = 0

        # 计算生成每个label的概率,labels_token_prob_list中所有token的概率相乘
        # 计算每个label的log prob
        log_probs = torch.sum(log_labels_token_prob_list, dim=-1)

        # 这里log_probs,概率的分布并不等于1,由于我们只是对所有可能的概率中取了最高概率的句子
        # action_sample
        log_total = torch.logsumexp(log_probs, dim=0)
        normalized_log_probs = log_probs - log_total
        if self.args.strategy == 'sample':
            dist = Categorical(probs=torch.exp(normalized_log_probs))
            action = dist.sample()
            log_prob = dist.log_prob(action)

        elif self.args.strategy == 'greedy':
            action = torch.argmax(normalized_log_probs)
            log_prob = normalized_log_probs[action] + 1e-8

        return action.item(), log_prob.item()
        # return action, log_prob


    def forward_value(self, encodings):
        outputs = self.value_model(**encodings)  # decoder输入为空，只考虑编码state return_dict=True, output_hidden_states=True
        if self.args.v_strategy == "mean":
            state_embedding = outputs.last_hidden_state.mean(dim=1)  # Shape: [batch_size, hidden_size]
        elif self.args.v_strategy == "max":
            state_embedding, _ = outputs.last_hidden_state.max(dim=1)
        else:
            state_embedding = outputs.last_hidden_state[:, -1, :]
        values = self.value_head.forward(state_embedding)
        return values  # 目前这里的values是batch的value


    def evaluation_actions(self, s: list, s_: list, candidate_action_texts: list, actions):
        input_text, output_text, dist_id = [], [], []
        for i, candidate_action_text in enumerate(candidate_action_texts):
            input_text.extend([s[i]] * len(candidate_action_text))
            output_text.extend(candidate_action_text)
            dist_id.extend([i] * len(candidate_action_text))
        dist_id = torch.tensor(dist_id, device=self.args.device)
        encodings = self.tokenizer(input_text, max_length=512, return_tensors="pt", padding="longest", truncation=True).to(self.args.device)
        encodings = {k: v for k, v in encodings.items()}
        # 将target_text转换为t5模型的输出格式
        labels = self.tokenizer(output_text, return_tensors="pt", max_length=512, padding="longest", truncation=True)['input_ids']
        # 由labels生成decoder_input_ids，需要在前面补0使得长度与labels相同
        decoder_input_ids = torch.cat([torch.zeros_like(labels[:, :1]), labels[:, :-1]], dim=-1).to(self.args.device)

        new_log_probs, new_entropys = self.batch_forward_policy(encodings, labels, decoder_input_ids, dist_id, len(s), actions)

        # get values  encodings['input_ids'][0]
        # old state values
        value_encodings = self.tokenizer(s, max_length=512, return_tensors="pt", padding="longest", truncation=True).to(self.args.device)
        value_encodings = {k: v for k, v in value_encodings.items()}
        values = self.forward_value(value_encodings)

        value_encodings_ = self.tokenizer(s_, max_length=512, return_tensors="pt", padding="longest", truncation=True).to(self.args.device)
        value_encodings_ = {k: v for k, v in value_encodings_.items()}
        values_ = self.forward_value(value_encodings_)

        return new_log_probs, new_entropys, values, values_

    def batch_forward_policy(self, encodings, labels, decoder_input_ids, dist_id, batch_size, actions):
        # 计算生成text的概率
        outputs = self.policy_model(**encodings, decoder_input_ids=decoder_input_ids)
        # 使用logits计算生成labels的概率，logits的shape为[batch_size,seq_len,vocab_size]
        logits = outputs["logits"]  # torch.Size([6, 7, 32128])
        # 计算log prob
        log_logits =F.log_softmax(logits, dim=-1)  # torch.Size([6, 7, 32128])
        # 选择log prob
        seq_len = torch.arange(labels.shape[-1]).expand(labels.shape[0], -1).to(self.args.device)  # torch.Size([6, 17])
        log_labels_token_prob_list = log_logits[torch.arange(labels.shape[0]).unsqueeze(1), seq_len, labels]
        # 将 padding 位置的 log prob 设置为 0
        log_labels_token_prob_list[labels == 0] = 0
        # 计算生成每个label的概率,labels_token_prob_list中所有token的概率相乘
        # 计算每个label的log prob
        log_probs = torch.sum(log_labels_token_prob_list, dim=-1, keepdim=True)  # torch.Size([6, 1])

        # 上面得到带梯度的当前执行action的log_prob
        log_prob_lst, entropy_lst = [], []
        for id in range(batch_size):
            idx = dist_id == id
            log_total = torch.logsumexp(log_probs[idx], dim=0)  # torch.Size([1])
            normalized_log_probs = (log_probs[idx] - log_total).reshape(1, -1)  # torch.Size([num_of_actions_for_id, 1])
            dist = Categorical(probs=torch.exp(normalized_log_probs))
            log_prob_lst.append(dist.log_prob(actions[id]))
            entropy_lst.append(dist.entropy())

        new_log_prob = torch.stack(log_prob_lst)
        new_entropy = torch.stack(entropy_lst)

        return new_log_prob, new_entropy
    

        # # 上面得到带梯度的当前执行action的log_prob
        # log_total = torch.logsumexp(log_probs, dim=0)  # torch.Size([1])
        # normalized_log_probs = (log_probs - log_total).reshape(1, -1)  # torch.Size([6, 1])
        # dist = Categorical(probs=torch.exp(normalized_log_probs))
        # log_prob = dist.log_prob(action)
        # entropy = dist.entropy()
    





