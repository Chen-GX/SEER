import math
import numpy as np
import copy

def score_aligned_entail_tree_proof(etree, prediction, gold_list, angle, gold_json_record:dict, pred_json_record:dict, bleurt_scorer=None, bleurt_buffer=None):
    """
    prediction: 预测的proof（str)
    gold_list: gold proof list(str)
    angle: proof
    gold_json_record: 原始信息 train.jsonl
    """
    sentences_pred, inferences_pred, int_to_all_ancestors_pred, relevant_sentences_pred, id_to_int_pred = parse_entailment_step_proof(prediction, gold_json_record=gold_json_record, slot_json_record=pred_json_record)
    sentences_gold, inferences_gold, int_to_all_ancestors_gold, relevant_sentences_gold, id_to_int_gold = parse_entailment_step_proof(gold_list[0], gold_json_record=gold_json_record, slot_json_record=gold_json_record)
    # pred_int_to_gold_int_mapping: 预测的int 和 gold int的对应关系
    # prediction_to_aligned_gold: 预测的int_str 和 gold_str的对应关系
    # prediction_to_perfect_match: 预测的每一个int_str 是否完美配对 True
    pred_int_to_gold_int_mapping, prediction_to_aligned_gold,  prediction_to_perfect_match = align_conclusions_across_proofs(int_to_all_ancestors_pred, int_to_all_ancestors_gold, id_to_int_pred, id_to_int_gold)

    # 统计在pred_key_tree上的int
    pred_key_tree_int = [item['int'] for item in int_to_all_ancestors_pred]

    # 下一个状态来自于tree结构的父子节点关系
    # 先从subtree中提取，每个int他的父节点是谁
    ancestor = {}
    for k, v in etree.subtree.items():
        ancestor[k] = None  # 初始化每个int没有父节点
        for child in v:
            if 'int' in child:
                ancestor[child] = k

    etree.int_ancestor = ancestor
    key_int_id, key_state_list, state_rewards  = [], [], []
    for state in etree.state_list[:-1]:
        cur_state = copy.deepcopy(state)
        if state.inter_con_id in pred_key_tree_int:  # 只关注关键的state   and state.intermediate_conclusion in prediction_to_perfect_match.keys()
            # 将关键节点的next_state换为tree上的节点
            if ancestor[state.inter_con_id] is not None:
                father_state_idx = int(ancestor[state.inter_con_id].replace('int', ''))
                cur_state.s_ = etree.state_list[father_state_idx - 1].s
                assert etree.state_list[father_state_idx - 1].inter_con_id == ancestor[state.inter_con_id]
            key_state_list.append(cur_state)
            key_int_id.append(cur_state.inter_con_id)

            if prediction_to_perfect_match[cur_state.inter_con_id]:
                state_rewards.append(etree.args.reward_value)
            else:
                state_rewards.append(etree.args.penalize_value)
        else:  # 多余的节点
            key_state_list.append(cur_state)
            key_int_id.append(cur_state.inter_con_id)
            state_rewards.append(etree.args.redundant_value)

    return state_rewards, key_state_list

def parse_entailment_step_proof(proof: str, gold_json_record: dict, slot_json_record=None):
    sentences = []
    inferences = []
    int_to_all_ancestors = dict()
    int_to_all_ancestors_list = []
    relevant_sentences = set()
    id_to_int = dict()

    if "[STEP]" in proof:
        proof_steps = proof.split('[STEP] ', maxsplit=-1)
    else:
        proof_steps = proof.split(';', maxsplit=-1)

    for p_step in proof_steps:
        # print(f"step:{p_step}")
        p_parts = p_step.strip().split(":", 1) # 只划分一次
        # print(f"step_parts:{p_parts}")

        step = p_parts[0]  # 'sent6 & sent12 -> int4'
        int_str = ""
        if len(p_parts) == 2:
            # int_str = normalize_sentence(p_parts[1].strip())
            int_str = p_parts[1].strip()  # 中间结论的文本

        if step:  # step A & B -> C
            # normalize t by numerically sorting sentence ids in LHS
            step_parts = step.split(' -> ', 1)
            if len(step_parts) == 2:
                rhs = step_parts[1].strip()
                # 一定保证int和state中的int 一模一样
                if len(int_str) == 0:
                    int_str = slot_json_record['meta']['triples'][rhs]
                elif int_str != slot_json_record['meta']['triples'][rhs]:
                    int_str = slot_json_record['meta']['triples'][rhs]

                if '&' in step_parts[0]:
                    lhs_ids = step_parts[0].split('&')
                else:
                    lhs_ids = step_parts[0].split(',')
                all_ancestors = set()
                lhs_ids = [lid.strip() for lid in lhs_ids]
                #print(f"\t for rhs={rhs}")
                for lid in lhs_ids:
                    if 'sent' in lid:
                        relevant_sentences.add(lid)
                        all_ancestors.add(lid)
                        #print(f"\t adding ancestor={lid}")
                    else:
                        their_ancestors = int_to_all_ancestors.get(lid, set())
                        all_ancestors = all_ancestors.union(their_ancestors)
                        #print(f"\t adding ancestors={their_ancestors}")

                sorted_lhs_ids = sorted(lhs_ids)
                sorted_lhs = ' & '.join(sorted_lhs_ids)
                sentences.append(f"{sorted_lhs} -> {rhs}")
                # print(f"lhs_ids:{lhs_ids}\t rhs = {rhs}\t all_ancestors={all_ancestors}")

                if rhs == "hypothesis":
                    # from utils.entail_trees_utils import normalize_sentence
                    int_str = normalize_sentence(gold_json_record['hypothesis'])
                # print(f"\t rhs = {rhs}, int_str={int_str}")

                id_to_int[rhs] = int_str
                int_to_all_ancestors[rhs] = all_ancestors
                int_to_all_ancestors_list.append(
                    {"int": rhs,
                    "ancestors":list(all_ancestors)
                     })
                inferences.append({
                    "lhs": sorted_lhs_ids,
                    "rhs": rhs
                })
    # print(f"\t<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    # print(f"\tproof:{proof}")
    # print(f"\tsentences:{sentences}")
    # print(f"\tinferences:{inferences}")
    # print(f"\tint_to_all_ancestors_list:{int_to_all_ancestors_list}")
    # print(f"\trelevant_sentences:{relevant_sentences}")
    # print(f"\tid_to_int:{id_to_int}")
    # sentences: list 每个推理步骤，A & B -> C
    # inferences list 推理的前后件 lsh: [A, V], rhs: C
    # int_to_all_ancestors_list: list 中间结论的父节点 {'int': C, 'ancestors': [A, B]}
    # relevant_sentences: dict 相关的叶子节点  {A, B}
    # id_to_int: dict {C: int_str}
    return sentences, inferences, int_to_all_ancestors_list, relevant_sentences, id_to_int

def normalize_sentence(sent):
    return sent.replace("  ", " ").replace(".", "").replace('\n', '').replace("( ", "").replace(" )", "").lower().strip()


def calculate_accumulated_rewards(each_step_reward, discount):
    n = len(each_step_reward)
    acc_stepwise_reward = [0] * n
    acc_stepwise_reward[-1] = each_step_reward[-1]
    for i in reversed(range(n - 1)):
        acc_stepwise_reward[i] = each_step_reward[i] + discount * acc_stepwise_reward[i + 1]
    return acc_stepwise_reward


def tree_reward_dfs(new_scores, node, subtree, init_int2score, gamma):
    if node not in subtree:   # 叶子节点且为'sent'节点
        return 0
    else:                     # 'int'节点
        if node in new_scores:  # 如果已经计算过，直接返回
            return new_scores[node]
        
        total_score = init_int2score[node]  # 开始为原始分数
        for sub_node in subtree[node]:     
            if 'int' in sub_node:         # 只有'int'类型节点需要计算
                total_score += tree_reward_dfs(new_scores, sub_node, subtree, init_int2score, gamma) * gamma
        new_scores[node] = total_score     # 更新新的分数
        return total_score


def parse_entailment_step_proof_remove_ids(proof: str, slot_json_record: dict):
    sentences = []
    inferences = []
    int_to_all_ancestors = dict()
    int_to_all_ancestor_ids = dict()
    int_to_all_ancestors_list = []
    relevant_sentences = set()
    id_to_int = dict()
    id_to_sentence = slot_json_record['meta']['triples']
    # print(f"PROOF:{proof}")

    if "[STEP]" in proof:
        temp = proof.split('[STEP] ', maxsplit=-1)
    else:
        temp = proof.split(';', maxsplit=-1)

    for t in temp:
        # 'sent1 & sent2 & sent3 & sent4 & sent7 -> int1: iron oxide is made of two elements: fe and o.
        t_parts = t.strip().split(":", 1) # 只划分一次
        t = t_parts[0]  # 'sent1 & sent9 -> int1'
        int_str = ""
        if len(t_parts) == 2:
            int_str = t_parts[1].strip()  # int1的结论'northern hemisphere receives the most direct sunlight in summer.'

        if t:
            # normalize t by numerically sorting sentence ids in LHS
            t_parts = t.split(' -> ', 1)  # 只对前面的拆分一次，这里倒不会出错
            if len(t_parts) == 2:
                rhs = t_parts[1].strip()
                # 一定保证int和state中的int 一模一样
                if len(int_str) == 0:
                    int_str = slot_json_record['meta']['triples'][rhs]
                elif int_str != slot_json_record['meta']['triples'][rhs]:
                    int_str = slot_json_record['meta']['triples'][rhs]
                    
                if '&' in t_parts[0]:
                    lhs_ids = t_parts[0].split('&')
                else:
                    lhs_ids = t_parts[0].split(',')
                all_ancestors = set()
                all_ancestor_ids = set()
                lhs_ids = [lid.strip() for lid in lhs_ids]
                lhs_strs = [id_to_sentence.get(lid.strip(), "NULL") for lid in lhs_ids]
                #print(f"\t for rhs={rhs}")
                for lid in lhs_ids:
                    if 'sent' in lid:
                        # print(f"\t adding ancestor={lid}\tid_to_sentence:{id_to_sentence}")
                        l_sent = id_to_sentence.get(lid, 'NULL')
                        relevant_sentences.add(l_sent)
                        all_ancestor_ids.add(lid)
                        all_ancestors.add(l_sent)
                    else:
                        their_ancestor_ids = int_to_all_ancestor_ids.get(lid, set())
                        all_ancestor_ids = all_ancestor_ids.union(their_ancestor_ids)

                        their_ancestors = int_to_all_ancestors.get(lid, set())
                        all_ancestors = all_ancestors.union(their_ancestors)

                        #print(f"\t adding ancestors={their_ancestors}")

                # sorted_lhs_ids = sorted(lhs_ids)
                sorted_lhs_ids = sorted(lhs_strs)
                sorted_lhs = ' & '.join(sorted_lhs_ids)

                if rhs == "hypothesis":
                    int_str = slot_json_record['hypothesis']

                # sentences.append(f"{sorted_lhs} -> {rhs}")
                sentences.append(f"{sorted_lhs} -> {int_str}")  # 不再是id，是句子的推理

                id_to_int[rhs] = int_str
                id_to_sentence[rhs] = int_str
                int_to_all_ancestor_ids[rhs] = all_ancestor_ids
                int_to_all_ancestors[rhs] = all_ancestors

                int_to_all_ancestors_list.append(
                    {"int": rhs,
                    "ancestors":list(all_ancestors),
                    "ancestor_ids":list(all_ancestor_ids)
                     })
                inferences.append({
                    "lhs": sorted_lhs_ids,
                    "rhs": int_str
                })

    return sentences, inferences, int_to_all_ancestors_list, relevant_sentences, id_to_int


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / max(1, union)

def align_conclusions_across_proofs(int_to_all_ancestors_pred: list, int_to_all_ancestors_gold: list, id_to_int_pred: dict, id_to_int_gold: dict):
    pred_int_to_gold_int_mapping = dict()
    prediction_to_aligned_gold = dict()
    prediction_to_perfect_match = dict()

    for pred_int_json in int_to_all_ancestors_pred:
        pred_int = pred_int_json['int']
        pred_ancestors = pred_int_json['ancestors']  # 对比句子
        prediction = id_to_int_pred[pred_int]  # 中间结论的text

        max_sim = 0
        best_gold_int = ""
        for gold_int_json in int_to_all_ancestors_gold:
            gold_int = gold_int_json['int']
            gold_ancestors = gold_int_json['ancestors']

            jaccard_sim = jaccard_similarity(pred_ancestors, gold_ancestors)

            if jaccard_sim > max_sim:
                max_sim = jaccard_sim
                best_gold_int = gold_int

        # if max_sim == math.isclose(max_sim, 1.0):
        if math.isclose(max_sim, 1.0):
            prediction_to_perfect_match[pred_int] = True
        else:
            prediction_to_perfect_match[pred_int] = False

        if best_gold_int:
            pred_int_to_gold_int_mapping[pred_int] = best_gold_int
            prediction_to_aligned_gold[prediction] = id_to_int_gold[best_gold_int]
        else:
            pred_int_to_gold_int_mapping[pred_int] = "NO_MATCH"
            prediction_to_aligned_gold[prediction] = ""

    return pred_int_to_gold_int_mapping, prediction_to_aligned_gold, prediction_to_perfect_match