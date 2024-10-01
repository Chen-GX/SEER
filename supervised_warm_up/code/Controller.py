import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # ignore TF log
import sys
os.chdir(sys.path[0])
import random
import copy
import os.path as osp
import glog as log
import json
import math
import time
import argparse
from collections import defaultdict
from itertools import combinations

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import PhrasalConstraint


from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist
import torch.backends.cudnn as cudnn

import bleurt 
from bleurt import score
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices(device_type='GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# from tree_utils import *
from exp_utils import create_optimizer, create_scheduler
from utils import Action, chunk

##### hrx experiment utils
import socket
import getpass
def get_random_dir_name():
    import string
    from datetime import datetime
    dirname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    vocab = string.ascii_uppercase + string.ascii_lowercase + string.digits
    dirname = dirname + '-' + ''.join(random.choice(vocab) for _ in range(8))
    return dirname

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

def set_log_file(fname, file_only=False):
    if file_only:
        log.logger.handlers[0].stream = log.handler.stream = sys.stdout = sys.stderr = open(fname, 'w', buffering=1)
    else:
        import subprocess
        tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
        os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
        os.dup2(tee.stdin.fileno(), sys.stderr.fileno())
#####


class Controller_dataset(Dataset):

    def __init__(self, data_path, loading_type = None, balance_sampling = False):
        
        datas = []
        datas_by_source = {}
        if type(data_path) == str:
            datas = [json.loads(line) for line in open(data_path).readlines()]
            balance_sampling = False
        elif type(data_path) == list and len(data_path) == 1:
            data_path = data_path[0]
            datas = [json.loads(line) for line in open(data_path).readlines()]
            balance_sampling = False
        else:
            for p_ in data_path:
                partial_datas = [json.loads(line) for line in open(p_).readlines()]
                datas += partial_datas
                datas_by_source[p_] = partial_datas
        
        # if loading_type == 'reason_only_pre':  # 这里不用处理
        #     for data_item in datas:
        #         if data_item['tgt_action']['type'] == Action.reason and 'con_sent' in data_item['tgt_action']['step']:
        #             del data_item['tgt_action']['step']['con_sent']
        #             data_item['tgt'] = Action.linearize_action(data_item['tgt_action'])

        self.datas = datas
        self.datas_by_source = datas_by_source
        self.loading_type = loading_type
        self.balance_sampling = balance_sampling

        print(f"{self.__class__.__name__} Loading from: {data_path}")
        print(f"Length of data: {len(self.datas)}")
        print({k:len(v) for k,v in self.datas_by_source.items()})
        print(f"balance_sampling: {balance_sampling}")
        

            
    def __getitem__(self, index):
        if self.balance_sampling == True:
            assert len(self.datas_by_source) > 1
            random_source = random.choice(list(self.datas_by_source.keys()))
            random_sample = random.choice(self.datas_by_source[random_source])
            return random_sample
        
        else:
            return self.datas[index]
    
    def __len__(self):
        return len(self.datas)


def train_one_step(batch, model, tokenizer, args):
    r"""
    train the model one step with the given batch data
    return the loss
    """
    model.train()
    
    # process batch data
    input_sents = [item['src'] for item in batch]
    output_sents = [item['tgt'] for item in batch]

    assert args.max_src_length == 512
    input_batch = tokenizer(
            input_sents,
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length', # 'longest',
            max_length=args.max_src_length,
            truncation=True,)

    output_batch = tokenizer(
                output_sents,
                add_special_tokens=True,
                return_tensors='pt',
                padding= 'max_length', # 'longest',
                max_length=args.max_tgt_length,
                truncation=True,)

    # Replace the <pad> to -100 for computing loss
    label_batch = output_batch['input_ids']
    label_batch.masked_fill_(label_batch == tokenizer.pad_token_id, -100) 
    
    input_batch['labels'] = label_batch
    input_batch = input_batch.to(model.device)
    
    # forward
    model_return = model(**input_batch)

    return model_return['loss']


def eval_model(model, data_loader, tokenizer, args):
    model.eval()

    inputs = []
    golds = []
    preds = []

    scores = []

    for batch in data_loader:

        # process batch data
        input_sents = [item['src'] for item in batch]
        output_sents = [item['tgt'] for item in batch]

        input_batch = tokenizer(
                input_sents,
                add_special_tokens=True,
                return_tensors='pt',
                padding='longest',
                max_length=args.max_src_length,
                truncation=True,
            )
        input_batch = input_batch.to(model.device)
        
        # generate
        generated = model.generate(
            input_ids = input_batch['input_ids'],
            attention_mask = input_batch['attention_mask'],
            top_p = 0.9,
            do_sample = True,
            max_length= args.max_tgt_length, 
            num_return_sequences = 1,
        )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

        inputs += input_sents
        golds += output_sents
        preds += decoded

        for pred_, gold_item in zip(decoded, batch):
            golds_ = gold_item['golds']
            scores.append({
                'acc': int(' '.join(pred_.split()) in golds_),
                'compare': (pred_, golds_),
            })

    eval_info = {
        'inputs': inputs,
        'golds': golds,
        'preds': preds,
        'scores': scores,
    }

    average_scores = {}
    for k in scores[0].keys():
        try:
            average_scores[k] = sum([s[k] for s in scores]) / len(scores)
        except:
            continue
    eval_info['average_scores'] = average_scores

    return average_scores['acc'], eval_info

def load_controller(exp_dir, model_name = 'best_model.pth'):
    print(f"Loading model from {osp.join(exp_dir, model_name)}")
    # read config
    config = json.load(open(osp.join(exp_dir,'config.json')))
    model_config = json.load(open(osp.join(exp_dir,'model.config.json')))
    args = argparse.Namespace(**config)

    # load model
    try:
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    except:
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    model.config.update(model_config)

    # load trained parameters
    state_dict = torch.load(osp.join(exp_dir, model_name), map_location='cpu')
    model.load_state_dict(state_dict)
    
    return model, tokenizer, args


def predict_controller(model, tokenizer, datas, bs = 4, mode = None, generate_args = {}):

    model.eval()
    # torch.cuda.empty_cache()
    
    generate_args['max_length'] = 128
    generate_args['return_dict_in_generate'] = True
    # generate_args['num_return_sequences'] = 1
    
    if mode is None or mode == 'sample':
        generate_args['do_sample'] = True
        if 'top_p' not in generate_args:
            generate_args['top_p'] = 0.9
        
    elif mode == 'beam_search':
        generate_args['do_sample'] = False
        generate_args['early_stopping'] = True
        generate_args['output_scores'] = True
        assert 'num_beams' in generate_args
        
    elif mode == 'constrained_beam_search':
        generate_args['do_sample'] = False
        generate_args['early_stopping'] = True
        generate_args['output_scores'] = True
        assert 'num_beams' in generate_args
        assert 'constraints' in generate_args
        
    else:
        raise NotImplementedError
    
        
    inputs = []
    preds = []
    preds_scores = []
    
    for batch in chunk(datas, bs):
        
        # process batch data
        input_sents = [item['src'] for item in batch]

        input_batch = tokenizer(
                input_sents,
                add_special_tokens=True,
                return_tensors='pt',
                padding='longest',
                max_length=512,
                truncation=True,)
        
        input_batch = input_batch.to(model.device)

        # generate
        output = model.generate(
            input_ids = input_batch['input_ids'],
            attention_mask = input_batch['attention_mask'],
            **generate_args,
        )

        decoded = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        
        assert len(decoded) % len(input_sents) == 0
        k = len(decoded) // len(input_sents) 
        decoded = [decoded[i * k : (i + 1) * k] for i in range(len(input_sents))]

        inputs += input_sents
        preds += decoded
        
        if generate_args.get('output_scores', False):
            output_scores = output.sequences_scores.detach().exp().cpu().numpy()
            output_scores = [output_scores[i * k : (i + 1) * k] for i in range(len(input_sents))]
            preds_scores += output_scores

    return preds, preds_scores


class Controller():
    def __init__(self, exp_dir, model_name = 'best_model.pth',  device='cuda'):
        
        # load model
        model,tokenizer,args = self.load_controller(exp_dir, model_name)
        model = model.to(device)
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.device = device

    def load_controller(self, exp_dir, model_name):
        # read config
        config = json.load(open(osp.join(exp_dir,'config.json')))
        model_config = json.load(open(osp.join(exp_dir,'model.config.json')))
        args = argparse.Namespace(**config)

        # load model
        print(f"Loading model from {exp_dir} {model_name}")
        if args.model_name_or_path in ['t5-large','t5-base','t5-small']:
            try:
                model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, cache_dir='/home/chenguoxin/workspace/FAME_Model')
            except:
                model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, local_files_only=True, cache_dir='/home/chenguoxin/workspace/FAME_Model')
            tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, local_files_only=True, cache_dir='/home/chenguoxin/workspace/FAME_Model')
        else:
            raise NotImplementedError

        model.config.update(model_config)

        # load trained parameters
        state_dict = torch.load(osp.join(exp_dir, model_name),map_location='cpu')
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

        return model,tokenizer,args

    def seq2seq_generate(self, input_sents, bs = 4, mode = None, generate_args = {}):

        model, tokenizer = self.model, self.tokenizer
        
        model.eval()

        generate_args['max_length'] = 128
        generate_args['num_return_sequences'] = generate_args.get('num_return_sequences', 1)
        generate_args['return_dict_in_generate'] = True

        if mode is None:
            mode = 'beam_search'

        if mode == 'beam_search':
            generate_args['do_sample'] = False
            generate_args['early_stopping'] = True
            generate_args['output_scores'] = True
            assert 'num_beams' in generate_args

        elif mode == 'constrained_beam_search':
            generate_args['do_sample'] = False
            generate_args['early_stopping'] = True
            generate_args['output_scores'] = True
            assert 'num_beams' in generate_args
            assert 'constraints' in generate_args

        # elif mode == 'sample':
        #     # sample mode do not support output_scores!
        #     # 'SampleEncoderDecoderOutput' object has no attribute 'sequences_scores'
        #     generate_args['do_sample'] = True
        #     generate_args['top_p'] = generate_args.get('top_p', 0.9)

        else:
            raise NotImplementedError


        inputs = []
        preds = []

        for batch_input_sents in chunk(input_sents, bs):

            input_batch = tokenizer(
                    batch_input_sents,
                    add_special_tokens=True,
                    return_tensors='pt',
                    padding='longest',
                    max_length=512,
                    truncation=True,)

            input_batch = input_batch.to(model.device)

            # generate
            output = model.generate(
                input_ids = input_batch['input_ids'],
                attention_mask = input_batch['attention_mask'],
                **generate_args,
            )

            decoded = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)

            assert len(decoded) % len(batch_input_sents) == 0
            k = len(decoded) // len(batch_input_sents) 
            decoded = [decoded[i * k : (i + 1) * k] for i in range(len(batch_input_sents))]

            inputs += batch_input_sents

            if generate_args.get('output_scores', False):
                # beam score
                output_sequences_scores = output.sequences_scores.detach().exp().cpu().numpy().tolist()
                output_sequences_scores = [output_sequences_scores[i * k : (i + 1) * k] for i in range(len(batch_input_sents))]

                # first token score
                if output.beam_indices is not None:
                    first_token_beam_scores = output.scores[0]
                    first_token_beam_scores = first_token_beam_scores.softmax(dim=1)
                    first_token_ids = output.sequences[:, 1] # [:,0] is <pad>
                    first_token_beam_indices = output.beam_indices[:,1] 

                    first_token_scores = []
                    for beam_i, token_id in zip(first_token_beam_indices, first_token_ids):
                        first_token_scores.append(float(first_token_beam_scores[beam_i,token_id].cpu()))
                    first_token_scores = [first_token_scores[i * k : (i + 1) * k] for i in range(len(batch_input_sents))]
                else:
                    first_token_scores = output_sequences_scores

                # merge
                for text, sequence_score, first_token_score in zip(decoded, output_sequences_scores, first_token_scores):
                    preds += [{
                        'text':text,
                        'sequence_score': sequence_score,
                        'first_token_score': first_token_score,
                    }]

            else:
                preds += [{'text':pred} for pred in decoded]

        # preds [len(input_sents), num_return_sequences]
        return preds

    def compute_ppl(self, input_sents, output_sents):
        
        model, tokenizer = self.model, self.tokenizer

        model.eval()

        assert len(input_sents) == len(output_sents)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id) # ignore <pad> !!!

        input_batch = tokenizer(input_sents,add_special_tokens=True,return_tensors='pt',padding='longest',truncation=True)
        output_batch = tokenizer(output_sents,add_special_tokens=True,return_tensors='pt', padding='longest',truncation=True)
        input_batch['labels'] = output_batch['input_ids']
        input_batch = input_batch.to(model.device)

        with torch.no_grad():
            logits = model(**input_batch)['logits']

        ppls = []
        for idx in range(len(output_sents)):
            loss = loss_fn(logits[idx].view(-1, logits.shape[-1]), input_batch['labels'][idx].view(-1))
            ppl = torch.exp(loss)
            ppls.append(float(ppl.cpu()))

        return ppls

    def compute_proved_score(self, input_sents, tau = 5.0):

        output_proved = [Action.linearize_action({'type':Action.end, 'is_proved': 'proved'})] * len(input_sents)
        output_unproved = [Action.linearize_action({'type':Action.end, 'is_proved': 'unproved'})] * len(input_sents)

        ppls = self.compute_ppl(input_sents*2, output_proved+output_unproved) # [sample_1_proved, ..., sample_n_proved, sample_1_unproved, ..., sample_n_unproved]
        ppls = torch.tensor(ppls).view(2, len(input_sents))

        scores = tau * 1.0/ppls # ‘1.0/ppls’ equals to the beam search sequence_scores, range 0~1; tau to scale the scores before softmax
        scores = scores.softmax(dim=0)
        proved_scores = scores[0, :].tolist()

        return proved_scores

    def __call__(self, *args, **kwargs):
        return self.seq2seq_generate(*args, **kwargs)



def run(args):

    # set random seed before init model
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(args.seed)

    if args.parallel == False:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dist.init_process_group(backend="nccl") 
        rank = dist.get_rank()
        print(f"Start running DDP on rank {rank}.")
        device = rank % torch.cuda.device_count()

    log.info("Loading data")
    train_dataset = Controller_dataset(args.train_data, loading_type=args.loading_type, balance_sampling=args.balance_sampling)
    log.info(f"Length of training dataest: {len(train_dataset)}")

    if args.parallel == False:
        train_loader = DataLoader(dataset = train_dataset,
                                    batch_size = args.bs,
                                    shuffle = True,
                                    num_workers = 4,
                                    collate_fn = lambda batch: batch)
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(dataset = train_dataset, 
                                    batch_size = args.bs // torch.cuda.device_count(), 
                                    num_workers = 4, 
                                    pin_memory = True, 
                                    collate_fn = lambda batch: batch,
                                    sampler=train_sampler)

    log.info(f"number of iteration each epoch : {len(train_loader)}")
    args.eval_iter = round(args.eval_epoch * len(train_loader))
    args.report_iter = round(args.report_epoch * len(train_loader))
    args.num_training_steps = args.epochs * len(train_loader)


    log.info("loading model")
    if args.model_name_or_path in ['t5-large','t5-base','t5-small']:
        try:
            model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, local_files_only=True, cache_dir=args.ptm)
            tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, local_files_only=True, cache_dir=args.ptm)
        except:
            model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, local_files_only=True)
            tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    else:
        raise NotImplementedError
    
    if args.resume_path:
        state_dict = torch.load(args.resume_path, map_location='cpu')
        model.load_state_dict(state_dict)
        log.info(f"Resume model parameters from {args.resume_path}")

    if args.local_rank <= 0:
        with open(osp.join(args.exp_dir, 'model.config.json'), 'w') as f:
            json.dump(vars(model.config), f, sort_keys=False, indent=4)


    if args.parallel == False:
        model = model.to(device)
    else:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.empty_cache()
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                        device_ids=[args.local_rank],
                                                        output_device=args.local_rank)

    optimizer = create_optimizer(model, args)
    lr_scheduler = create_scheduler(optimizer, args)
    

    log.info("start training")
    global_iter = 0
    loss_list = []
    best_metric = -100

    for epoch_i in range(1, args.epochs+1):
        
        if args.parallel: train_sampler.set_epoch(epoch_i)

        for batch in train_loader:
            loss = train_one_step(batch,model,tokenizer,args)
            
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            
            optimizer.zero_grad()
            
            global_iter += 1
            
            if args.local_rank <= 0:
                
                if not global_iter % args.report_iter:
                    log.info(f"Epoch {global_iter/len(train_loader):.1f} training loss {np.mean(loss_list):.4f}")
                    loss_list = []
                else:
                    loss_list.append(float(loss.cpu().data))

                # save checkpoint
                if (not global_iter % args.eval_iter) and (args.save_model):
                    save_path = osp.join(args.exp_dir,f'step{global_iter}_model.pth')
                    if args.parallel == False:
                        torch.save(model.state_dict(), save_path)
                    else:
                        torch.save(model.module.state_dict(), save_path)
                    log.info(f"Iteration {global_iter} save checkpoint model")
                
        log.info(f"Epoch {epoch_i} finished")

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument("--ptm", type=str, default="../../exp/ptm") 
    # dateset
    parser.add_argument("--task", type=str, default="task_2")
    parser.add_argument("--data_dir", type=str, default="../preprocess_data")
    parser.add_argument("--train_data", type=str, default="../preprocess_data")  
    parser.add_argument('--dev_data', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument("--loading_type", type=str, default="reason_only_pre") 
    parser.add_argument("--balance_sampling", action='store_true', default=False)

    
    # model
    parser.add_argument("--model_name_or_path", type=str, default="t5-large", help="")  
    parser.add_argument("--resume_path", type=str, default="", help="")                
    parser.add_argument('--max_src_length', type=int, default=512, )
    parser.add_argument('--max_tgt_length', type=int, default=128, )

    # optimization
    parser.add_argument('--bs', type=int, default=16, help='input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train')
                        
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adafactor', action='store_true', default=False)
    parser.add_argument('--lr_scheduler_type', type=str, default='constant_with_warmup')
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--accumulation_steps', type=int, default=1)

    parser.add_argument('--eval_epoch', type=float, default=1.0)

    # seed
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # exp and log
    parser.add_argument("--exp_dir", type=str, default='../result/test')
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--report_epoch', type=float, default=1.0)
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    args.train_data = []
    for d_split in ['train']:
        args.train_data.append(osp.join(args.data_dir, args.task, f"{d_split}.jsonl"))

    args.exp_dir = osp.join(args.exp_dir, args.task)
    
    return args


if __name__ == '__main__':
    args = get_params()

    # check parallel training
    if torch.cuda.device_count() > 1:
        args.parallel = True
    else:
        args.parallel = False

    if args.local_rank <= 0: # -1 for single gpu; 0 for the first gpu
        if args.seed == 0:
            args.seed = random.randint(1,1e4)

        args.exp_dir = osp.join(args.exp_dir, get_random_dir_name())

        os.makedirs(args.exp_dir, exist_ok=True)
        set_log_file(osp.join(args.exp_dir, 'run.log'), file_only=True)
        
        # make metrics.json for logging metrics
        args.metric_file = osp.join(args.exp_dir, 'metrics.json')
        open(args.metric_file, 'a').close()
        
        os.makedirs(osp.join(args.exp_dir, 'prediction'), exist_ok=True)



        # dump config.json
        with open(osp.join(args.exp_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)

        # backup scripts
        os.system(f'cp -r ../code {args.exp_dir}')

        log.info('Host: {}, user: {}, CUDA_VISIBLE_DEVICES: {}, cwd: {}'.format(
            socket.gethostname(), getpass.getuser(), os.environ.get('CUDA_VISIBLE_DEVICES', ''), os.getcwd()))
        log.info('Python info: {}'.format(os.popen('which python').read().strip()))
        log.info('Command line is: {}'.format(' '.join(sys.argv)))
        log.info('Called with args:')
        print_args(args)

        run(args)

        # make 'done' file
        open(osp.join(args.exp_dir, 'done'), 'a').close()

    else:
        run(args)