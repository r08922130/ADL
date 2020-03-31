import numpy as np
import torch
import torch.nn as nn 
import sys
from model import S2S
import os
from solver import Solver
import json
import pickle
from dataset import SeqTaggingDataset
from dataset import Seq2SeqDataset
from postprocessing import Postprocessing
from utils import Tokenizer
import preprocess_seq2seq

if __name__ == "__main__":
    
    arg = sys.argv
    solver = Solver()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg[1] == "train":
        #python src/main.py train batch_size
        with open("datasets/seq2seq/train.pkl", 'rb') as f:
            train = pickle.load(f)
        with open("datasets/seq2seq/valid.pkl", 'rb') as f:
            valid = pickle.load(f)
        with open("datasets/seq2seq/embedding.pkl", 'rb') as f:
            embedding = pickle.load(f)
        
        batch_size = int(arg[2])
        t_l = len(train)
        if t_l%batch_size==0:    
            t_bl = t_l//batch_size
        else:
            t_bl = t_l//batch_size+1
        v_l = len(valid)
        if v_l%batch_size==0:
            v_bl = v_l//batch_size
        else:
            v_bl = v_l//batch_size+1
        train_batches = [train.collate_fn([train[j] for j in range(i*batch_size,min((i+1)*batch_size,t_l))]) for i in range(t_bl)]
        #print(batches['text'][0:batch_size])
        #train_data = batches['text']
        #train_label = batches['label']
        #train_key = batches['key']
        #train_sent = batches['sent_range']

        valid_batches = [valid.collate_fn([valid[j] for j in range(i*batch_size,min((i+1)*batch_size,v_l))]) for i in range(v_bl)]
        #print(batches['text'][0:batch_size])
        #valid_data = batches['text']
        #valid_label = batches['label']
        #print(train_data[0])
        #print(train_label[0])
        #valid_key = batches['key']
        #valid_sent = batches['sent_range']
        emb_w = embedding.vectors
        vocab = embedding.vocab
        attention = True if arg[4] == 'A' else False
        mymodel = S2S(emb_w.size(0),emb_w.size(1),256,len(emb_w),device,layer=2,attention=attention).to(device)
        mymodel.embedding.from_pretrained(emb_w)
        """if arg[4] == 'pre':
            mymodel.load_state_dict(torch.load("ckpt/best.ckpt"))"""
        solver.train(mymodel,train_batches,valid_batches,attention=attention,batch_size=batch_size,device=device,epoch=int(arg[3]))
    else:
        with open("datasets/seq2seq/embedding.pkl", 'rb') as f:
            embedding = pickle.load(f)
        emb_w = embedding.vectors
        vocab = embedding.vocab
        print(len(vocab))
        attention = True if arg[3] == 'A' else False
        if arg[1] == 'valid':
            #valid
            with open("datasets/seq2seq/valid.pkl", 'rb') as f:
                data = pickle.load(f)
            mode = 'valid'
            
        else:
            #test
            if arg[2] == 'TA':
                with open(arg[1]) as f:
                    test = [json.loads(line) for line in f]
                with open( 'datasets/seq2seq/config.json') as f:
                    config = json.load(f)
                tokenizer = Tokenizer(lower=config['lower_case'])
                tokenizer.set_vocab(embedding.vocab)
                data = preprocess_seq2seq.create_seq2seq_dataset_without_save(
                    preprocess_seq2seq.process_samples(tokenizer, test),
                    config,tokenizer.pad_token_id)
            else:
                with open("datasets/seq2seq/test.pkl", 'rb') as f:
                    data = pickle.load(f)
            
            mode = 'test'
        batch_size = int(arg[6])
        l = len(data) 
        if l%batch_size==0:    
            bl = l//batch_size
        else:
            bl = l//batch_size+1
        
        data_batches = [data.collate_fn([data[j] for j in range(i*batch_size,min((i+1)*batch_size,l))]) for i in range(bl)]
        print(data_batches[0]['text'])
        
        
        mymodel = S2S(emb_w.size(0),emb_w.size(1),256,len(emb_w),device,layer=2,attention=attention).to(device)
        mymodel.embedding.from_pretrained(emb_w)
        if os.path.isfile(arg[4]):
            mymodel.load_state_dict(torch.load(arg[4],map_location= device))
        #solver.test
        result = solver.test(mymodel,data_batches,device,mode=mode)
        post = Postprocessing()
        dict_result = []
        dict_result = post.indiesToSentences(result,dict_result,vocab,tokenizer,mode=mode)
        post.toJson(arg[5],dict_result)  
            