import os, sys
os.chdir(sys.path[0])
import os.path as osp
import json
import random
import numpy as np
import torch
import re

from itertools import combinations

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
    print(f"Random seed set as {seed}")

def decapitalize(sent):
    if len(sent) > 1:
        return sent[0].lower() + sent[1:]
    else:
        return sent.lower()

def normalize(sent):
    """
    add period to a sentence, and decapitalize
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

def add_fullstop(sent):
    if sent.endswith('.'):
        return sent
    else:
        return sent+'.'

def linearize_context(sent_list = None, sent2id=None):
    s = ""

    sentX_sents = []
    not_sentX_sents = []
    for sent in sent_list:
        if sent2id[sent].startswith('sent'):
            sentX_sents.append(sent)
        else:
            not_sentX_sents.append(sent)

    for sent in not_sentX_sents + sentX_sents:
        s += f"{sent2id[sent]}: {add_fullstop(sent)} "
    
    return s

class prepare_warm_data:
    def __init__(self, step_proof_dir, save_step_reward_dir):
        self.step_proof_dir = step_proof_dir
        self.save_step_reward_dir = save_step_reward_dir
        self.data_split = ['train', 'dev', 'test']
        self.state_pattern = "$question$ {Q} $answer$ {A} $hypothesis$ {H} $proof$ {proof} $context$ {context}"  # S

    def preprocess(self):
        for d_split in self.data_split:
            data = []
            with open(osp.join(self.step_proof_dir, f'{d_split}.jsonl'), 'r', encoding='utf-8') as f:
                items = json.load(f)
                
                for item in items:
                    id2sent = item['sentences']
                    sent2id = {}
                    for k, v in id2sent.items():
                        sent2id[v] = k
                    context = linearize_context(list(id2sent.values()), sent2id)
                    
                    input_text = self.state_pattern.format(
                        Q = item['question'],
                        A = item['answer'],
                        H = add_fullstop(item['hypothesis']),
                        proof = "; ".join(item['current_proof']),
                        context = context
                    )
                    # print(input_text)
                    output_text = f"reason: {' & '.join(item['chosen'])}"
                    # print(output_text)
                    data.append({'src': input_text.strip(), 'tgt': output_text})
                

            print(f'{d_split} len {len(data)}')
            # store the data file
            with open(osp.join(self.save_step_reward_dir, f'{d_split}.jsonl'), 'w', encoding='utf-8') as f:
                for item in data:
                    json.dump(item, f)
                    f.write("\n")
                
                


if __name__ == "__main__":
    set_seed()
    step_proof_dir = "./step_reward_data/task_1"
    save_warm_data_dir = "./task_1"
    os.makedirs(step_proof_dir, exist_ok=True)
    os.makedirs(save_warm_data_dir, exist_ok=True)

    process_data = prepare_warm_data(step_proof_dir, save_warm_data_dir)
    process_data.preprocess()