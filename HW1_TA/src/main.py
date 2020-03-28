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
        batch_size = int(arg[2])
        batches = train.collate_fn(train)
        print(batches['text'][0:batch_size])
    