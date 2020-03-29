import numpy as np
import torch
import torch.nn as nn 
import sys
from model import SequenceTaggle
from model import SequenceTaggle1
import os
from solver import Solver
import json
import pickle
from dataset import SeqTaggingDataset
if __name__ == "__main__":
    
    arg = sys.argv
    solver = Solver()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg[1] == "train":
        #python src/main.py train batch_size
        with open("datasets/seq_tag/train.pkl", 'rb') as f:
            train = pickle.load(f)
        with open("datasets/seq_tag/valid.pkl", 'rb') as f:
            valid = pickle.load(f)
        with open("datasets/seq_tag/embedding.pkl", 'rb') as f:
            embedding = pickle.load(f)
        
        batch_size = int(arg[2])
        
        batches = train.collate_fn(train)
        #print(batches['text'][0:batch_size])
        train_data = batches['text']
        train_label = batches['label']
        #train_key = batches['key']
        #train_sent = batches['sent_range']

        batches = valid.collate_fn(train)
        #print(batches['text'][0:batch_size])
        valid_data = batches['text']
        valid_label = batches['label']
        print(train_data.size())
        #valid_key = batches['key']
        #valid_sent = batches['sent_range']
        emb_w = embedding.vectors
        mymodel = SequenceTaggle(emb_w.size(0),emb_w.size(1),256,1,device,layer=4).to(device)
        mymodel.embedding.from_pretrained(emb_w)
        """if arg[4] == 'pre':
            mymodel.load_state_dict(torch.load("ckpt/best.ckpt"))"""
        solver.train(mymodel,train_data,train_label,valid_data,valid_label,batch_size=batch_size,device=device,epoch=int(arg[3]))
    