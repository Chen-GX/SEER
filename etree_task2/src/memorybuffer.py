import torch
import random
import numpy as np
import math

class MemoryBuffer(object):
    def __init__(self, args) -> None:
        # self.args = args
        self.k_epochs = args.k_epochs
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.s, self.s_, self.done = None, None, None
        self.candidate_action_text, self.action= None, None
        self.log_prob = None
        self.len_ = None
        self.rewards = None
        self.reset()
        
    def reset(self):
        self.s = []  # 当前的状态的输入（文本）
        self.s_ = []  # 下一步的状态
        self.done = []
        self.action = []  # 当前状态采用的action的idx
        self.candidate_action_text = []  # 当前状态所有的action
        self.rewards = []
        self.log_prob = []
        self.len_ = 0

    def add(self, reward, state):
        self.s.append(state.s)
        self.s_.append(state.s_)
        self.done.append(state.done)  # True or False
        self.action.append(state.action)  # int
        self.candidate_action_text.append(state.candidate_action_text)
        self.log_prob.append(state.log_prob)
        self.rewards.append(reward)

    def not_empty(self):
        return len(self.s) > 0

    def is_full(self):
        return len(self.s) == self.buffer_size

    def get_len_(self):
        self.len_ = len(self.s)
        return self.len_

    def get_order(self, batch_size):
        # 从存储的数据中按batch_size返回一个batch
        start_index = 0
        while start_index < len(self.s):
            end_index = start_index + batch_size
            if end_index > len(self.s):
                end_index = len(self.s)
            
            yield {
                's': self.s[start_index: end_index],
                's_': self.s_[start_index: end_index],
                'done': self.done[start_index: end_index],
                'action_text': self.action_text[start_index: end_index],
                'action': self.action[start_index: end_index],
                'log_prob': self.log_prob[start_index: end_index],
                'rewards': self.rewards[start_index: end_index],
            }
            
            start_index = end_index

    def sample_sequentially(self):
        for i in range(0, len(self.s), self.batch_size):
            states = self.s[i: i+self.batch_size]
            next_states = self.s_[i: i+self.batch_size]
            dones = self.done[i: i+self.batch_size]
            actions = self.action[i: i+self.batch_size]
            candidate_action_text = self.candidate_action_text[i: i+self.batch_size]
            log_probs = self.log_prob[i: i+self.batch_size]
            rewards = self.rewards[i: i+self.batch_size]
            yield states, next_states, dones, actions, candidate_action_text, log_probs, rewards

    def get(self, batch_size):
        # 创建一个包含所有键的列表
        keys = ['s', 's_', 'done', 'action_text', 'action', 'candidate_action_text', 'log_prob', 'rewards']
        # 使用字典推导式创建一个映射，将键映射到相应的存储列表
        dict_buffer = {key: getattr(self, key) for key in keys}

        indices = list(range(len(self.s)))
        random.shuffle(indices)

        while indices:
            if len(indices) < batch_size:
                batch_indices = indices
                indices = []
            else:
                batch_indices, indices = indices[:batch_size], indices[batch_size:]

            yield {key: [dict_buffer[key][i] for i in batch_indices] for key in keys}
    


# class RolloutBuffer(object):
#     def __init__(self, args) -> None:
#         self.args = args
#         self.buffer_size = args.buffer_size
#         self.n_envs = 1
#         self.s, self.s_, self.done = None, None, None
#         self.action_text, self.action_index, self.values, self.entropy = None, None, None, None
#         self.log_prob = None
#         self.count = 0
#         self.reset()

#     def reset(self):
#         self.s = np.array([])  # 将其它的数据也转换为numpy数组
#         self.s_ = np.array([])
#         self.done = np.array([])
#         self.action_text = np.array([])
#         self.action = np.array([])
#         self.rewards = np.array([])
#         self.values = np.array([])
#         self.entropy = np.array([])
#         self.log_prob = np.array([])

#     def add(self, buffer_info):
#         self.s = np.append(self.s, buffer_info['s'])
#         self.s_ = np.append(self.s_, buffer_info['s_'])
#         self.done = np.append(self.done, buffer_info['done'])
#         self.action_text = np.append(self.action_text, buffer_info['action_text'])
#         self.action = np.append(self.action, buffer_info['action'])
#         self.values = np.append(self.values, buffer_info['values'])
#         self.entropy = np.append(self.entropy, buffer_info['entropy'])
#         self.log_prob = np.append(self.log_prob, buffer_info['log_prob'])

#     def get(self, batch_size):
#         indices = list(range(len(self.s)))
#         random.shuffle(indices)

#         while indices:
#             if len(indices) < batch_size:
#                 batch_indices = indices
#                 indices = []
#             else:
#                 batch_indices = [indices.pop() for _ in range(batch_size)]

#             yield {
#                 's': self.s[batch_indices],
#                 's_': self.s_[batch_indices],
#                 'done': self.done[batch_indices],
#                 'action_text': self.action_text[batch_indices],
#                 'action': self.action[batch_indices],
#                 'values': self.values[batch_indices],
#                 'entropy': self.entropy[batch_indices],
#                 'log_prob': self.log_prob[batch_indices],
#                 'rewards': self.rewards[batch_indices],
#             }