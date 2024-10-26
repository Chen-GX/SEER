import os, sys
os.chdir(sys.path[0])
import os.path as osp
import json
from collections import Counter

def load_jsonl(file):
    data = []
    with open(file, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip())['id'])
    return data


def check_data_and_slots():
    """检查data和slots之间的task1, 2, 3是否id一样"""
    slots_path = '/home/chenguoxin/reflexion/LLM_baseline/React_etree/entailment_bank_get_metric/data/processed_data/slots'
    data_path = '/home/chenguoxin/reflexion/exp/data/entailment_trees_emnlp2021_data_v3/dataset'


    task_levels = ['task_1', 'task_2', 'task_3']
    for i in range(len(task_levels)):
        for data_split in ['train', 'dev', 'test']:
            slots_path1 = osp.join(slots_path, f'{task_levels[i]}-slots', f'{data_split}.jsonl')
            data_path1 = osp.join(data_path, task_levels[i], f'{data_split}.jsonl')

            slots = load_jsonl(slots_path1)
            data = load_jsonl(data_path1)
            if set(slots) != set(data):
                
                print(Counter(slots))
                print("".center(20, "*"))
                print(Counter(data))
                print(data_split)
                print()
            for id_i, id_j in zip(slots, data):
                if id_i != id_j:
                    print(data_split, ":", id_i, id_j)


def check_task123():
    """检查task1, 2, 3之间是否id一样"""
    slots_path = '/home/chenguoxin/reflexion/LLM_baseline/React_etree/entailment_bank_get_metric/data/processed_data/slots'
    data_path = '/home/chenguoxin/reflexion/exp/data/entailment_trees_emnlp2021_data_v3/dataset'

    for root_path in [slots_path, data_path]:
        if slots_path == root_path:
            task_levels = ['task_1-slots', 'task_2-slots', 'task_3-slots']
        else:
            task_levels = ['task_1', 'task_2', 'task_3']
        for i in range(len(task_levels)):
            for j in range(i + 1, len(task_levels)):
                for data_split in ['train', 'dev', 'test']:
                    path1 = osp.join(root_path, task_levels[i], f'{data_split}.jsonl')
                    path2 = osp.join(root_path, task_levels[j], f'{data_split}.jsonl')

                    data1 = load_jsonl(path1)
                    data2 = load_jsonl(path2)
                    if set(data1) != set(data2):
                        
                        print(Counter(data1))
                        print("".center(20, "*"))
                        print(Counter(data2))
                        print(root_path, '\n', task_levels[i], task_levels[j], '\n', data_split)
                        print()
                    for id_i, id_j in zip(data1, data2):
                        if id_i != id_j:
                            print(root_path, '\n', task_levels[i], task_levels[j], '\n', data_split, ":", id_i, id_j)


def check_slots_angles():
    """检查slots和angles之间id是否一样"""
    slots_path = '/home/chenguoxin/reflexion/LLM_baseline/React_etree/entailment_bank_get_metric/data/processed_data/slots'
    angles_path = '/home/chenguoxin/reflexion/LLM_baseline/React_etree/entailment_bank_get_metric/data/processed_data/angles'

    task_levels = ['task_1', 'task_2', 'task_3']
    for i in range(len(task_levels)):
        for data_split in ['train', 'dev', 'test']:
            slots_path1 = osp.join(slots_path, f'{task_levels[i]}-slots', f'{data_split}.jsonl')
            angles_path1 = osp.join(angles_path, task_levels[i], f'{data_split}.jsonl')

            slots = load_jsonl(slots_path1)
            angles = load_jsonl(angles_path1)
            if set(slots) != set(angles):
                
                print(Counter(slots))
                print("".center(20, "*"))
                print(Counter(angles))
                print(task_levels[i], data_split)
                print()
            for id_i, id_j in zip(slots, angles):
                if id_i != id_j:
                    print(data_split, ":", id_i, id_j)

check_task123()
check_data_and_slots()
check_slots_angles()