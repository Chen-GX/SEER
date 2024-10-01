import json
import numpy as np
import os
    
### -------- DR --------
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
# ref: https://colab.research.google.com/github/UKPLab/sentence-transformers/blob/master/examples/applications/retrieve_rerank/retrieve_rerank_simple_wikipedia.ipynb#scrollTo=0rueR6ovrs01

class Dense_Retriever():
    def __init__(self, corpus, encoder_model, device='cuda', buffer_file = None):
        """
        dense retriever based on the cos sim of embedding
        Speed Optimization:  normalize_embeddings + dot_score = cos_sim
        """
        
        assert type(corpus) == dict
        
        corpus_text = list(corpus.values())

        print("Converting corpus to embedding")
        corpus_embeddings = encoder_model.encode(corpus_text, batch_size = 32, 
                                                 convert_to_tensor=True, show_progress_bar=True)
        corpus_embeddings = corpus_embeddings.to(device)
        corpus_embeddings = util.normalize_embeddings(corpus_embeddings) 
        
        buffer = {}
        if buffer_file:
            try:
                print(f"Retriever buffer file: {buffer_file}")
                if os.path.exists(buffer_file):
                    with open(buffer_file) as f:
                        # fcntl.flock(f, fcntl.LOCK_EX)
                        buffer = json.load(f)
                    print(f"Load buffer, length: {len(buffer)}")
            except:
                print(f"Retriever buffer error")
                buffer_file = None
                
        
        self.corpus = corpus
        self.corpus_text = corpus_text
        self.encoder_model = encoder_model
        self.corpus_embeddings = corpus_embeddings
        self.device = device
        self.buffer_file = buffer_file
        self.buffer = buffer
        self.last_buffer_len = len(buffer)
        self.retrieve_top_n = 25
        
    def search(self, query, n = 5):

        queries = [query] if type(query) != list else query
        queries = self.query_normalization(queries)
        
        query_embeddings = self.encoder_model.encode(queries, convert_to_tensor=True, show_progress_bar = False)
        query_embeddings = query_embeddings.to(self.device)
        query_embeddings = util.normalize_embeddings(query_embeddings)
        
        top_k = max(n, 100) # by default, we return the top-100 result
        hits = util.semantic_search(query_embeddings, self.corpus_embeddings, top_k=top_k, score_function=util.dot_score) 
        
        # post-process result
        for i, h in enumerate(hits):
            for item in h:
                item['text'] = self.corpus_text[item['corpus_id']]
                item['index'] = item['corpus_id']
            hits[i] = h
        
        # write buffer
        if self.buffer_file:
            self.write_buffer(queries, hits)
        
        # top-n
        hits = [h[:n] for h in hits]
        
        return hits[0] if type(query) != list else hits
            
    def search_with_buffer(self, query, n = 5):
        
        queries = [query] if type(query) != list else query
        queries = self.query_normalization(queries)  # 小写
        
        hits = [[] for _ in queries]
        
        index_not_in_buffer = []
        for index, q in enumerate(queries):
            if q in self.buffer and len(self.buffer[q]) >= n:
                hits[index] = self.buffer[q][:n]
            else:
                index_not_in_buffer.append(index)
                
        if len(index_not_in_buffer) > 0:
            search_hits = self.search([queries[index] for index in index_not_in_buffer], n = n)
            for search_i, index in enumerate(index_not_in_buffer):
                hits[index] = search_hits[search_i]
        
        assert all([len(h) > 0 for h in hits])
        
        return hits[0] if type(query) != list else hits
    
    def write_buffer(self, queries, hits):
        for q, h in zip(queries, hits):
            self.buffer[q] = h
        
        # save buffer to file
        if len(self.buffer) - self.last_buffer_len > 100:
            with open(self.buffer_file, 'w') as f:
                # fcntl.flock(f, fcntl.LOCK_EX) # lock the file
                json.dump(self.buffer, f)
            self.last_buffer_len = len(self.buffer)
        
        
    def query_normalization(self, queries):
        normalized_queries = []
        for query in queries:
            nq = query.lower()
            normalized_queries.append(nq)
        return normalized_queries
        
    
    def __call__(self, query, n=5):
        return self.search_with_buffer(query, n)
        