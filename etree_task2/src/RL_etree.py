import numpy as np
import copy
import time
from RL_state import State
import logging
logger = logging.getLogger(__name__)

class eTree(object):
    """控制当前etree的推理"""
    def __init__(self, args, data_item, agent) -> None:
        self.args = args
        self.data_item = data_item  # 初始数据
        # self.get_gold_tree()
        self.wrong_node_reward = -1
        # 存储这棵etree的推理结果
        self.proof = None
        # agent 负责更新状态
        self.agent = agent
        # 构建etree
        self.whole_id2sent = {}
        self.whole_sent2id = {}
        self.subtree = {}

        # 存储state
        self.state_list = [self.init_root_state()]

    def init_root_state(self):
        # 根据data_item 初始化 root state
        root_state = State(self.args)
        root_state.init_from_data_item(self.data_item)
        self.worldtree_provenance = {}
        for k, v in root_state.id2sent.items():
            self.whole_id2sent[k] = v
            self.whole_sent2id[v] = k
            self.worldtree_provenance[k] = {"original_text": v}

        return root_state

    def get_next_state(self, state, action_name):
        # 推理得到最优的intermediate conclusion
        sents = [state.id2sent[a] for a in action_name]
        best_con, best_score = self.agent.do_entail(sents, state)
        if best_score == -1:  # 避免best_score全是负1的情况，概率比较小
            if self.args.random_ts:
                timestamp = time.strftime("%H_%M_%S",time.localtime())
                best_con += f'_{state.num_int + 1}_{timestamp}'
            logger.info(f'all -1 score')
        # 更新当前state的信息
        state.chosen = action_name
        state.inter_con_id = f'int{state.num_int + 1}'
        # 更新并维护全局的句子到id的字典
        state.step = f"{' & '.join(action_name)} -> int{state.num_int + 1}: {best_con}"
        state.intermediate_conclusion = best_con
        # state.intermediate_bleurt_score = self.agent.bleurt_scorer.score(references=[state.H], candidates=[state.intermediate_conclusion])[0]
        state.intermediate_bleurt_score = self.agent.get_bleurt_score(state.H, [state.intermediate_conclusion])[0]
        state.proof_str.append(" & ".join(sorted(action_name)) + f" -> {best_con}")

        # 构建etree
        self.subtree[state.inter_con_id] = state.chosen
        self.whole_id2sent[state.inter_con_id] = state.intermediate_conclusion
        self.whole_sent2id[state.intermediate_conclusion] = state.inter_con_id
        # 记录当前步骤，并扩展下一状态
        s_ = State(self.args)
        s_.copy_from_previous_state(state)
        
        # 什么时候结束（这里再看一下）
        done = self.evaluate_end(state, s_)
        
        # 更新PPO需要的参数
        state.s_ = s_.linearize_state(form=self.args.linearize_state_form)
        state.done = done
        return s_, done
    
    def evaluate_end(self, state, s_):
        if len(s_.id2sent.values()) == 1:
            done = True
        elif state.intermediate_bleurt_score > self.args.threshold:
            done = True
        else:
            done = False
        return done
    

    def construct_proof(self, root):
        part_proof = []
        choices = []
        premises = []
        if root in self.subtree.keys():
            part_proof.append(" & ".join(self.subtree[root]) + f" -> {root}: {self.whole_id2sent[root]}")
            choices.append(([self.whole_id2sent[sub] for sub in self.subtree[root]], self.whole_id2sent[root]))
            premises.extend(p for p in self.subtree[root] if p.startswith('sent'))
            for child in self.subtree[root]:
                c_proof, c_choice, c_p = self.construct_proof(child)
                if len(c_proof) > 0:
                    part_proof.extend(c_proof)
                    choices.extend(c_choice)
                    premises.extend(c_p)
        return part_proof, choices, premises
    
    def build_proof_task3(self):
        # 最后一个结论视为最优结论
        subtree = copy.deepcopy(self.subtree)
        if self.subtree:  # 如果字典不为空
            h_id, h = subtree.popitem()
            steps, choices, premises = self.construct_proof(h_id)
            # premises = list(dict.fromkeys(premises))
            steps.reverse()
            self.pred = "; ".join(steps)  # 这里没有逆转，重大bug
            # steps[-1] = steps[-1].split('->')[0] + '-> hypothesis;'  # 去掉这个
            self.choices = choices[::-1]
        else:
            steps = []
            self.pred = ""
            self.choices = []

        
        self.proof = "; ".join(steps)
        
        self.prediction_json = {
            'id': self.data_item['id'],
            'slots': {'proof': self.pred},
            'meta': {'triples': self.whole_id2sent},
            'hypothesis': self.data_item['hypothesis']
        }