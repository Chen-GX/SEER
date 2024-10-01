import itertools
import copy
import random
import logging
logger = logging.getLogger(__name__)

import spacy
spacy_nlp = spacy.load("en_core_web_sm")

from utils import normalize, sort_key, add_fullstop, sent_IoU, same_sent

class Action:
    retrieve = 'retrieve'
    reason = 'reason'
    end = 'end'

    @classmethod
    def linearize_action(cls, action):
        action_str = ""

        if action['type'] == Action.retrieve:
            action_str += f"{action['type']}: {action['query_id']}"

        elif action['type'] == Action.reason:
            action_str += f"{action['type']}: "
            action_str += " & ".join(sorted(action['step']['pre_id']))
            if 'con_sent' in action['step']:
                action_str += f" -> {action['step']['con_sent']}"

        elif action['type'] == Action.end:
            action_str += f"{action['type']}"
            if 'is_proved' in action:
                action_str += f": {action['is_proved']}"

        else:
            raise NotImplementedError

        return action_str

    @classmethod
    def parse_action(cls, action_str):

        if ':' in action_str:
            action_type, paras_str = action_str.split(':', maxsplit=1)
            action_type = action_type.strip()
        else:
            action_type = action_str.strip()
            paras_str = None

            
        if action_type == Action.retrieve:
            action = {
                'type': action_type,
                'query_id': paras_str.strip(),
            }
        
        elif action_type == Action.reason:
            if '->' not in paras_str:
                pre_id = [p.strip() for p in paras_str.split('&')]
                action = {
                    'type': action_type,
                    'step': {
                        'pre_id': pre_id,
                    },
                    'use_module': True,
                }
            else:
                pre_id_str, con_sent = paras_str.split('->', maxsplit=1)
                pre_id = [p.strip() for p in pre_id_str.split('&')]
                action = {
                    'type': action_type,
                    'step': {
                        'pre_id': pre_id,
                        'con_sent': con_sent.strip(),
                    },
                    'use_module': False,
                }

        elif action_type == Action.end:
            if paras_str is None:
                action = {
                    'type': action_type,
                }
            else:
                action = {
                    'type': action_type,
                    'is_proved': paras_str.strip(),
                }  
        else:
            action = None

        return action

class State(object):
    def __init__(self, args):
        self.args = args
        self.data_id = None
        self.Q = ''
        self.A = ''
        self.H = ''  # 此次推理树下的假设
        self.num_int = 0  # 中间结论的个数
        self.used_premises = {}  # 已经使用过的sent {sent: id}
        # 当前状态下，句子-id的映射
        self.sent2id = {}
        self.id2sent = {}

        self.context = ''  # 当前状态下的所有前提组成的context

        # take action后更新的参数
        self.chosen = []
        self.inter_con_id = None
        self.intermediate_conclusion = ''
        self.step = ''
        self.proof_str = []

        # PPO更新需要的参数
        self.s = None
        self.s_ = None
        self.action_text = None
        self.action = None
        self.candidate_action_text = None
        self.done = None
        self.log_prob = None

    def copy_from_previous_state(self, state):
        self.data_id = state.data_id
        self.H = state.H
        self.A = state.A
        self.Q = state.Q
        # 继承上一个状态的used_S
        self.used_premises = copy.deepcopy(state.used_premises)
        # 更新sent2id和id2sent，used_S
        for s_id, s_text in state.id2sent.items():
            if s_id in state.chosen:
                self.used_premises[s_text] = s_id
            else:
                self.sent2id[s_text] = s_id
                self.id2sent[s_id] = s_text
        self.num_int = state.num_int + 1
        self.id2sent[f"int{self.num_int}"] = state.intermediate_conclusion
        self.sent2id[state.intermediate_conclusion] = f"int{self.num_int}"
        # 根据sent2_id和id2sent，来更新P和ids，以及context
        self.context = " ".join([f"{k}: {v}" for k, v in sorted(self.id2sent.items(), key=sort_key)])
        self.proof_str = copy.deepcopy(state.proof_str)
        

    def init_from_data_item(self, data_item):
        self.data_id = data_item['id']
        self.H = normalize(data_item['hypothesis'])
        self.Q = normalize(data_item['question'])
        self.A = normalize(data_item['answer'])
        self.id2sent = copy.deepcopy(data_item['meta']['triples'])

        for k, v in self.id2sent.items():
            self.sent2id[v] = k
  
        self.context = " ".join([f"{k}: {v}" for k, v in sorted(self.id2sent.items(), key=sort_key)])

    def generate_candidate_action(self):
        # 这里如果self.ids 只有1个的话，他是没有办法采样出来的
        if len(self.ids) == 1:
            action_text = [f"reason: {self.ids[0]}"]
        else:
            self.candidate_action = list(itertools.combinations(self.ids, 2))
            n_samples = min(len(self.candidate_action), 5)
            sample_elements = random.sample(self.candidate_action, n_samples)  # 本身不允许重复采样
            # [(sent1, sent2)]
            action_text = [f"reason: {s1} & {s2}" for (s1, s2) in sample_elements]
        return action_text


    def linearize_proof(self):
        proof_str = "; ".join(self.proof_str)
        return proof_str

    def linearize_context(self, sent_list = None):
        s = ""

        sentX_sents = []
        not_sentX_sents = []
        for sent in sent_list:
            if self.sent2id[sent].startswith('sent'):
                sentX_sents.append(sent)
            else:
                not_sentX_sents.append(sent)

        for sent in not_sentX_sents + sentX_sents:
            s += f"{self.sent2id[sent]}: {add_fullstop(sent)} "
        
        return s

    def linearize_state(self, form = 'QAHPS'):
        
        if form == 'default' or form == 'HPS':
            state_str = ""
            state_str += f"$hypothesis$ {add_fullstop(self.H)} "
            state_str += f"$proof$ {self.linearize_proof()} "
            state_str += f"$context$ {self.linearize_context(list(self.id2sent.values()))} "
        elif form == 'QAHPS':
            state_str = ""
            state_str += f"$question$ {self.Q} $answer$ {add_fullstop(self.A.lower())} "
            state_str += f"$hypothesis$ {add_fullstop(self.H)} "
            state_str += f"$proof$ {self.linearize_proof()} "
            state_str += f"$context$ {self.linearize_context(list(self.id2sent.values()))}" # 这里会给句子加句号
        elif form == 'QAHPN':
            node_sents = []
            for step in self.P[::-1]: #  from root to leaves
                for sent in [step['con_sent']] + step['pre_sent']:
                    if sent not in node_sents:
                        node_sents.append(sent)
                        
            state_str = ""
            state_str += f"$question$ {self.Q} $answer$ {add_fullstop(self.A.lower())} "
            state_str += f"$hypothesis$ {add_fullstop(self.H)} "
            state_str += f"$proof$ {self.linearize_proof()} "
            state_str += f"$node$ {self.linearize_context(node_sents)} "     
        elif form == 'QACHPN':
            node_sents = []
            for step in self.P[::-1]: #  from root to leaves
                for sent in [step['con_sent']] + step['pre_sent']:
                    if sent not in node_sents:
                        node_sents.append(sent)
                        
            state_str = ""
            state_str += f"$question$ {self.Q} $answer$ {add_fullstop(self.A.lower())} "
            state_str += f"$choices$ {self.choices_str} "
            state_str += f"$hypothesis$ {add_fullstop(self.H)} "
            state_str += f"$proof$ {self.linearize_proof()} "
            state_str += f"$node$ {self.linearize_context(node_sents)} "     
        else:
            raise NotImplementedError
        
        return state_str

    def check_action_executable(self, action_str):  # 有用
        if action_str is None:
            return False

        # try to parse the str
        if type(action_str) == str:
            try:
                action = Action.parse_action(action_str)
            except Exception as e:
                # print(f"check_action_executable failed: {str(e)}. action_str: {action_str}")
                action = None

            if action is None:
                return False
        else:
            action = action_str
            if 'type' not in action:
                return False

        # check by action type
        if action['type'] == Action.retrieve:
            return False
            if 'query' in action:
                query = action['query']
            elif 'query_id' in action:
                # id2sent = {sent_id:sent for sent, sent_id in self.sent2id.items()}
                query = self.id2sent.get(action['query_id'], "")  # 这里不考虑检索hypothesis
                action['query'] = query
            else:
                return False

            if not query:
                return False

        elif action['type'] == Action.reason:
            if 'step' not in action:
                return False

            # check 'pre_id' / 'pre_sent'
            if 'pre_sent' in action['step']:
                pre_sent = action['step']['pre_sent']
            elif 'pre_id' in action['step']:
                # id2sent = {sent_id:sent for sent, sent_id in self.state.sent2id.items()}
                pre_sent = [self.id2sent.get(p, "") for p in action['step']['pre_id']]
                action['step']['pre_sent'] = pre_sent
            else:
                return False

            if any([sent not in self.id2sent.values() for sent in pre_sent]):
                # premise not in state.S
                return False
            if len(set(action['step']['pre_sent'])) != len(action['step']['pre_sent']):
                # premises: sent1 & sent1
                return False

            # # filter pre_sent by rules
            # if self.check_premise_overlap and False:
            #     # Rule: if the pre_sent have no overlap or have too much overlap, we reject the step            
            #     if len(action['step']['pre_sent']) < 2:
            #         return False
            #     pre_iou = max([sent_IoU(ps[0], ps[1], spacy_nlp) for ps in itertools.combinations(action['step']['pre_sent'], 2)])
            #     if pre_iou == 0.0 or pre_iou >= 0.7:
            #         # print("*** filter step ***", action['step']['pre_sent'])
            #         return False
            

            # check 'con_sent'
            if 'con_sent' not in action['step']:
                # use module to get con_sent
                pass
            else:
                return False
                con_sent = action['step']['con_sent']
                if any([same_sent(con_sent, s) for s in pre_sent]):
                    # conclusion repeats one of the premises 
                    return False
                if any([same_sent(con_sent, s) for s in self.state.used_S]):
                    # print('con_sent in used_S')
                    return False
                
                S_int = [sent for sent in self.state.S if self.state.sent2id[sent].startswith('int')]
                if any([same_sent(con_sent, s) for s in S_int]):
                    # some intermediate conclusions could be found as facts in corpus
                    # we only return false when the con_sent has been a int
                    # print('con_sent has been int')
                    return False


        elif action['type'] == Action.end:
            return False

        else:
            raise NotImplementedError
            
        return True
    
    def generate_legal_action(self):
        # 考虑reason
        if len(self.id2sent) == 1:  # 前提不足两个，别推理了
            action_text = ["end"]
        else:
            candidate_action = list(itertools.combinations(self.id2sent.keys(), 2))
            sample_elements = random.sample(candidate_action, min(len(candidate_action), 5))  # 本身不允许重复采样
            # [(sent1, sent2)]
            action_text = [f"reason: {s1} & {s2}" for (s1, s2) in sample_elements]
        return action_text
    
    def next_id(self, ident='int'):
        assert ident in ['sent', 'int']
        for i in itertools.count(1):
            if f"{ident}{i}" not in list(self.used_premises.keys()) + list(self.id2sent.keys()):
                return f"{ident}{i}"  




