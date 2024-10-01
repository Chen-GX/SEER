import os
import os.path as osp
import random
import numpy as np
import torch
import json
import re
import string
import itertools
import copy

import logging
logger = logging.getLogger(__name__)

def set_seed(seed: int = 1024) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set as {seed}")


def get_task1(args, train, data_split):
    # 得到task1
    data_path = osp.join(args.data_dir, 'task_1', f'{data_split}.jsonl')
    num = 0
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            id2sent = item['meta']['triples']  # 这里不深拷贝，让int放进triples中
            proofs = item['proof'].split(';')
            # gold_step = []
            for proof in proofs:
                proof = proof.strip().split(":")
                if len(proof[0]) > 0:  # proof的最后存在一个空行
                    entail = proof[0].split('->')
                    src, tgt = entail[0].strip(), entail[1].strip()
                    # sents = src.split('&')
                    # tmp_step = [id2sent[s.strip()] for s in sents]
                    if tgt.strip() == 'hypothesis':
                        # gold_step.append((tmp_step, item['hypothesis']))
                        id2sent[tgt.strip()] = item['hypothesis']
                    else:
                        # gold_step.append((tmp_step, proof[-1].strip()))
                        id2sent[tgt.strip()] = proof[-1].strip()
            # item['gold_step'] = gold_step
            # item['gold_id2sent'] = id2sent
            train[num]['task1'] = item
            num += 1

    return train

def get_task2(args, train, data_split):
    # 得到task1
    data_path = osp.join(args.data_dir, 'task_2', f'{data_split}.jsonl')
    num = 0
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            id2sent = item['meta']['triples']  # 这里不深拷贝，让int放进triples中
            proofs = item['proof'].split(';')
            # gold_step = []
            for proof in proofs:
                proof = proof.strip().split(":")
                if len(proof[0]) > 0:  # proof的最后存在一个空行
                    entail = proof[0].split('->')
                    src, tgt = entail[0].strip(), entail[1].strip()
                    # sents = src.split('&')
                    # tmp_step = [id2sent[s.strip()] for s in sents]
                    if tgt.strip() == 'hypothesis':
                        # gold_step.append((tmp_step, item['hypothesis']))
                        id2sent[tgt.strip()] = item['hypothesis']
                    else:
                        # gold_step.append((tmp_step, proof[-1].strip()))
                        id2sent[tgt.strip()] = proof[-1].strip()
            # item['gold_step'] = gold_step
            # item['gold_id2sent'] = id2sent
            train[num]['task2'] = item
            num += 1

    return train


def load_data(args, agent=None, return_dict=False):

    def do_retrieve(agent, item):
        # 进来的就是合法动作
        query = item['hypothesis']
        item['meta']['triples'] = {}
        retrieval_result = agent.retriever(query, n=agent.retriever.retrieve_top_n)

        # {'corpus_id': 7236, 'score': 0.7541006207466125, 'text': 'xxx', 'index': 7236}
        retrieval_result = [r['text'] for r in retrieval_result]
        # 重新分配id
        for sent in list(dict.fromkeys(retrieval_result)):  # 保持有序并且去重
            item['meta']['triples'][next_id('sent', item)] = sent

        return
    
    def next_id(ident='int', item=None):
        assert ident in ['sent', 'int']
        for i in itertools.count(1):
            if f"{ident}{i}" not in list(item['meta']['triples'].keys()):
                return f"{ident}{i}"
    
    total_data = {}
    for data_split in ['train', 'dev', 'test']:
        data_path = osp.join(args.data_dir, args.task, f'{data_split}.jsonl')
        datas = [json.loads(line) for line in open(data_path).readlines()]
        datas = get_task2(args, datas, data_split)
        # # 不做检索
        # for item in datas:
        #     do_retrieve(agent, item)
        total_data[data_split] = datas

    # total_data['train'] = get_task1(args, total_data['train'])
    if args.debug_max_num != 0:
        for data_split in ['train', 'dev', 'test']:
            total_data[data_split] = total_data[data_split][:args.debug_max_num]
            logger.info(f"Loading data from {data_path}: {len(total_data[data_split])}")


    if return_dict:
        return total_data
    return total_data['train'], total_data['dev'], total_data['test']

def aggregate_ancestor(gold):
    int2leaf = {}
    aggre_leaves = []
    for idx, step in enumerate(gold):
        # step_leaf = []  # 0: step source name; 1: aggregated leaf
        leaves = []
        inter_name = 'int' + str(idx + 1)

        for name in step[0]:
            if name[0] == 'i':
                leaves.extend(int2leaf[name][1])
            else:
                leaves.append(name)

        int2leaf[inter_name] = [list(step), list(set(leaves))]
        aggre_leaves.append(leaves)
    return aggre_leaves

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / max(1, union)

def remove_period(sent):
    """
    remove the period of a sentence
    """
    return sent.strip(string.punctuation)

# sentence process

def add_fullstop(sent):
    if sent.endswith('.'):
        return sent
    else:
        return sent+'.'

def decapitalize(sent):
    if len(sent) > 1:
        return sent[0].lower() + sent[1:]
    else:
        return sent.lower()

def normalize(sent):
    """
    add period to a sentence, and decapitalize  在句子中添加句号，并取消大写
    """
    if sent.endswith('.'):
        return decapitalize(sent).strip()
    else:
        return decapitalize(sent).strip() + '.'

def sort_key(item):
    match = re.match(r'(\D+)(\d+)', item[0])
    prefix = match.group(1)
    number = int(match.group(2))
    return (prefix, number)

def LCstring(string1,string2):
    len1 = len(string1)
    len2 = len(string2)
    res = [[0 for i in range(len1+1)] for j in range(len2+1)]
    result = 0
    for i in range(1,len2+1):
        for j in range(1,len1+1):
            if string2[i-1] == string1[j-1]:
                res[i][j] = res[i-1][j-1]+1
                result = max(result,res[i][j])  
    return result


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def same_sent(sent1, sent2):
    return normalize_answer(sent1) == normalize_answer(sent2)

def sent_IoU(sent1,sent2,spacy_nlp):
    sent1 = normalize_answer(sent1)
    sent2 = normalize_answer(sent2)
    
    spacy_nlp.Defaults.stop_words -= {"using", "show","become","make","down","made","across","put","see","move","part","used"}
    
    doc1 = spacy_nlp(sent1)
    doc2 = spacy_nlp(sent2)    
    
    word_set1 = set([token.lemma_ for token in doc1 if not (token.is_stop or token.is_punct)])
    word_set2 = set([token.lemma_ for token in doc2 if not (token.is_stop or token.is_punct)])
    
    inter = 0
    for word1 in word_set1:
        for word2 in word_set2:
            lcs = LCstring(word1,word2)
            if lcs / min(len(word1),len(word2)) > 0.6:
                inter += 1
                break
    # print(word_set1)
    # print(word_set2)
    
    iou = inter / (len(word_set1.union(word_set2))+1e-10)
    return iou


def chunk(it, n):
    # 它将一个可迭代对象（如列表）分割成多个大小为 n 的块。
    # 如果最后剩下的元素个数小于 n，那么这些元素还会形成一个额外的块。
    c = []
    for x in it:
        c.append(x)
        if len(c) == n:
            yield c
            c = []
    if len(c) > 0:
        yield c