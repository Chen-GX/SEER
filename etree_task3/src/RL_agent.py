import os
import os.path as osp
import torch
import json
import argparse
import numpy as np
import fcntl
import itertools

from bleurt import score
from transformers import T5ForConditionalGeneration, T5Tokenizer
from Retriever import Dense_Retriever
from sentence_transformers import SentenceTransformer


import logging
logger = logging.getLogger(__name__)

from utils import normalize_answer, chunk

class Agent(object):
    """控制推理"""
    def __init__(self, args) -> None:
        self.args = args

        # 各个模块
        self.entail_model, self.entail_tokenizer = self.load_entailment_module(exp_dir=self.args.entailment_module_exp_dir)  # 4G
        self.entail_prefixes = ['deductive substitution:', 'deductive conjunction:', 'deductive if-then:']
        self.entail_input_text_pattern = "{sents} $hypothesis$ {H}"
        self.generate_args = {"num_beams": 5, "num_return_sequences": 5}
        self.bs = 100
    
        # load buffer
        if args.use_buffer:
            buffer = {}
            if args.buffer_file:
                try:
                    logger.info(f"EntailmentModule buffer file: {args.buffer_file}")
                    if osp.exists(args.buffer_file):
                        with open(args.buffer_file, 'r') as f:
                            # fcntl.flock(f, fcntl.LOCK_EX)
                            buffer = json.load(f)
                        logger.info(f"Load buffer, length: {len(buffer)}")
                except:
                    logger.info(f"EntailmentModule buffer error")
                    # buffer_file = None
            self.buffer = buffer
            self.last_buffer_len = len(buffer)

        if args.use_bleurt_buffer:
            bleurt_buffer = {}
            if args.bleurt_buffer_file:
                try:
                    logger.info(f"bleurt buffer file: {args.bleurt_buffer_file}")
                    if osp.exists(args.bleurt_buffer_file):
                        with open(args.bleurt_buffer_file, 'r') as f:
                            # fcntl.flock(f, fcntl.LOCK_EX)
                            bleurt_buffer = json.load(f)  # key H + con: score
                        logger.info(f"Load bleurt buffer, length: {len(bleurt_buffer)}")
                except:
                    logger.info(f"bleurt buffer error")
                    # buffer_file = None
            self.bleurt_buffer = bleurt_buffer
            self.last_bleurt_buffer_len = len(bleurt_buffer)

        self.new_buffer_file_path()

        # bleurt 模型
        logger.info(f"Loading BLEURT model from {self.args.bleurt_path}")
        self.bleurt_scorer = score.BleurtScorer(self.args.bleurt_path)  # 参数是冻结的

        # Retriever
        logger.info(f"Loading corpus from {args.corpus_path}")
        corpus = json.load(open(args.corpus_path))
        logger.info(f"Loading Retriever from {args.retriever_path_or_name}")
        bi_encoder = SentenceTransformer(args.retriever_path_or_name)
        self.retriever = Dense_Retriever(corpus, bi_encoder, buffer_file=None, device=args.device)  # 将corpus编码为embed
        
        # 冻结模型参数
        self.frozen_net()

    def frozen_net(self):
        # 冻结entail module
        for param in self.entail_model.parameters():
            param.requires_grad = False

    def do_retrieve(self, action, state):
        # 进来的就是合法动作
        if 'query' in action:
            query = action['query']
        elif action['query_id'] == 'hypothesis':
            query = state.H
            action['query'] = query
        elif 'query_id' in action:
            query = state.id2sent.get(action['query_id'], "")
            action['query'] = query
        retrieval_result = self.retriever(query, n=self.retriever.retrieve_top_n)

        # {'corpus_id': 7236, 'score': 0.7541006207466125, 'text': 'xxx', 'index': 7236}
        retrieval_result = [r['text'] for r in retrieval_result]

        # 重新分配id
        for sent in list(dict.fromkeys(retrieval_result)):  # 保持有序并且去重
            state.id2sent[state.next_id('sent')] = sent

        return



    def load_entailment_module(self, exp_dir, model_name='best_model.pth'):
        # read config
        config = json.load(open(osp.join(exp_dir,'config.json')))
        model_config = json.load(open(osp.join(exp_dir,'model.config.json')))
        entail_args = argparse.Namespace(**config)

        # load model
        logging.info(f"Loading model from {exp_dir} {model_name}")
        if entail_args.model_name_or_path in ['t5-large','t5-base','t5-small']:
            model = T5ForConditionalGeneration.from_pretrained(entail_args.model_name_or_path, cache_dir=self.args.cache_dir, local_files_only=True)
            tokenizer = T5Tokenizer.from_pretrained(entail_args.model_name_or_path, cache_dir=self.args.cache_dir, local_files_only=True)
        else:
            raise NotImplementedError

        model.config.update(model_config)
        # load trained parameters
        state_dict = torch.load(osp.join(exp_dir, model_name))
        model.load_state_dict(state_dict)

        return model.to(self.args.no_grad_device), tokenizer


    def seq2seq_generate(self, input_sents, mode='beam_search', generate_args={}):  # 推理中间结论

        model, tokenizer = self.entail_model, self.entail_tokenizer
        
        model.eval()

        generate_args['max_length'] = 128
        generate_args['num_return_sequences'] = generate_args.get('num_return_sequences', 1)
        generate_args['return_dict_in_generate'] = True

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
        else:
            raise NotImplementedError


        inputs = []
        preds = []
        preds_scores = []

        for batch_input_sents in chunk(input_sents, self.bs):

            input_batch = tokenizer(
                    batch_input_sents,
                    add_special_tokens=True,
                    return_tensors='pt',
                    padding='longest',  # padding到此次batch的最大值
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

            inputs += batch_input_sents  # 输入
            preds += decoded  # 输出

            if generate_args.get('output_scores', False):
                output_scores = output.sequences_scores.detach().exp().cpu().numpy().tolist()
                output_scores = [output_scores[i * k : (i + 1) * k] for i in range(len(batch_input_sents))]
                preds_scores += output_scores

        if self.args.use_buffer:
            self.write_buffer(input_sents, preds, preds_scores)
        
        return preds, preds_scores

    def new_buffer_file_path(self):
        buffer_file_path, buffer_file_name = os.path.split(self.args.buffer_file)
        buffer_file_base, buffer_file_ext = os.path.splitext(buffer_file_name)
        os.makedirs(osp.join(buffer_file_path, "tmp"), exist_ok=True)
        self.this_buffer_file = osp.join(buffer_file_path, "tmp", buffer_file_base + '_' + self.args.timestamp + buffer_file_ext)

        bleurt_buffer_file_path, bleurt_buffer_file_name = os.path.split(self.args.bleurt_buffer_file)
        bleurt_buffer_file_base, bleurt_buffer_file_ext = os.path.splitext(bleurt_buffer_file_name)
        os.makedirs(osp.join(bleurt_buffer_file_path, "tmp"), exist_ok=True)
        self.this_bleurt_buffer_file = osp.join(bleurt_buffer_file_path, "tmp", bleurt_buffer_file_base + '_' + self.args.timestamp + bleurt_buffer_file_ext)


    def write_buffer(self, input_sents, preds, preds_scores):
        for input_sent, p, ps in zip(input_sents, preds, preds_scores):
            self.buffer[input_sent] = {'preds': p, 'preds_scores': ps}
        
        # save buffer to file
        if len(self.buffer) - self.last_buffer_len > self.args.write_buffer_step:
            with open(self.this_buffer_file, 'w') as f:
                fcntl.flock(f, fcntl.LOCK_EX) # lock the file
                json.dump(self.buffer, f, indent=4)
            self.last_buffer_len = len(self.buffer)
            logger.info(f'write to buffer')

    def write_bleurt_buffer(self, no_buffer_text, H, no_buffer_scores):
        for text, s in zip(no_buffer_text, no_buffer_scores):
            self.bleurt_buffer[f'{text} -> {H}'] = s
        
        # save buffer to file
        if len(self.bleurt_buffer) - self.last_bleurt_buffer_len > self.args.write_buffer_step:
            with open(self.this_bleurt_buffer_file, 'w') as f:
                fcntl.flock(f, fcntl.LOCK_EX) # lock the file
                json.dump(self.bleurt_buffer, f, indent=4)
            self.last_bleurt_buffer_len = len(self.bleurt_buffer)
            logger.info(f'write to bleurt buffer')

    def final_write_buffer(self):
        with open(self.this_buffer_file, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX) # lock the file
            json.dump(self.buffer, f, indent=4)
        self.last_buffer_len = len(self.buffer)
        logger.info(f'final write to buffer')
        
        # begin_time = time.time()
        with open(self.this_bleurt_buffer_file, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX) # lock the file
            json.dump(self.bleurt_buffer, f, indent=4)
        self.last_bleurt_buffer_len = len(self.bleurt_buffer)
        logger.info(f'fianl write to bleurt buffer')

    def generate_with_buffer(self, input_sents, generate_args = {}):

        num_return_sequences = generate_args.get('num_return_sequences', 1)

        preds = [[] for _ in input_sents]
        preds_scores = [[] for _ in input_sents]

        index_not_in_buffer = []
        if self.args.use_buffer:
            for index, input_sent in enumerate(input_sents):
                if input_sent in self.buffer:
                    br = self.buffer[input_sent]
                    if len(br['preds']) >= num_return_sequences:
                        br = {k:v[:num_return_sequences] for k,v in br.items()}
                        preds[index] = br['preds']
                        preds_scores[index] = br['preds_scores']
                    else:
                        index_not_in_buffer.append(index)
                else:
                    index_not_in_buffer.append(index)
        else:
            index_not_in_buffer = list(range(len(input_sents)))

        if len(index_not_in_buffer) > 0:
            rs = self.seq2seq_generate([input_sents[index] for index in index_not_in_buffer], generate_args = generate_args)
            for rs_i, index in enumerate(index_not_in_buffer):
                preds[index] = rs[0][rs_i]
                if generate_args.get('output_scores', False):
                    preds_scores[index] = rs[1][rs_i]
        
        return preds, preds_scores


    def get_reward_score(self, step_input):
        # step_input for ["xx", "xx"]
        # tokenizer
        reward_score = []
        encoding = self.reward_net.tokenizer(step_input, truncation=True, padding='longest', max_length=512, return_tensors='pt')
        encoding = encoding.to(self.args.no_grad_device)
        logits = self.reward_net.forward(**encoding).view(-1)
        reward_score = torch.sigmoid(logits).detach().cpu().numpy()
        return reward_score

    def get_bleurt_score(self, H, step_input):
        if self.args.use_bleurt_buffer:
            bleurt_scores = [None for _ in step_input]
            index_not_in_buffer = []
            for index, text in enumerate(step_input):
                key = f"{text} -> {H}"
                if key in self.bleurt_buffer:
                    bleurt_scores[index] = self.bleurt_buffer[key]
                else:
                    index_not_in_buffer.append(index)

            if len(index_not_in_buffer) > 0:
                no_buffer_text = [step_input[idx] for idx in index_not_in_buffer]
                no_buffer_scores = self.bleurt_scorer.score(references=[H] * len(no_buffer_text), candidates=no_buffer_text)
                for score_i, idx in enumerate(index_not_in_buffer):
                    bleurt_scores[idx] = no_buffer_scores[score_i]
                # write to file
                self.write_bleurt_buffer(no_buffer_text, H, no_buffer_scores)
        else:
            bleurt_scores = self.bleurt_scorer.score(references=[H] * len(step_input), candidates=step_input)
        return bleurt_scores

    def do_entail(self, sents, state):
        H = state.H
        # 基于action进行推理，并获得最优的con
        premise = self.entail_input_text_pattern.format(sents=" ".join(sents), H=H)

        # 组织entailment的输入形式
        prefixed_premise = [f"{prefix} {premise}" for prefix in self.entail_prefixes]
        preds, preds_scores = self.generate_with_buffer(prefixed_premise, generate_args=self.generate_args)
        preds = np.array(preds).flatten().tolist()
        preds_scores = np.array(preds_scores).flatten()
        # forbidden_cons = set(list(state.id2sent.values()) + list(state.used_premises.keys()))  # 禁止出现一样的结论
        forbidden_cons = list(dict.fromkeys(list(state.id2sent.values()) + list(state.used_premises.keys())))
        # # Preprocess all sentences
        normalized_preds = [normalize_answer(s) for s in preds]
        normalized_forbidden_cons = {normalize_answer(s) for s in forbidden_cons}
        # # Create a boolean mask where True means the prediction is in forbidden_cons
        normalized_is_forbidden = np.array([npred in normalized_forbidden_cons for npred in normalized_preds])
        preds_scores[normalized_is_forbidden] = -1
        # is_forbidden = np.array([pred in forbidden_cons for pred in preds])
        # preds_scores[is_forbidden] = -1

        # bleurt 分数
        # bleurt_score = np.array(self.get_bleurt_score(H, preds))  # 非常耗时

        # 取最优的分数
        # con_scores = np.mean([preds_scores, bleurt_score], axis=0)
        best_idx= np.argmax(preds_scores)
        best_con = preds[best_idx]
        return best_con, preds_scores[best_idx]