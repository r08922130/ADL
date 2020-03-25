import numpy as np
import torch
import torch.nn as nn 
import sys
from preprocessing import Preprocessing
from model import SequenceTaggle
import os
from solver import Solver
import json
if __name__ == "__main__":
    
    arg = sys.argv
    solver = Solver()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if arg[1] == '--train':
        print("Training")
        # cmd : python main.py --train emb_dim 30 
        dim = arg[2] if arg[2] else 50
        dim = int(dim)
        if not os.path.isfile("embedding.npy"):
            print("No File . Preprocessing!")
            pre = Preprocessing()
            pre.batch_data(dim=300,modes=['train','valid','test'])
        
        train_data = np.load("data/train_data_{}.npy".format(dim),allow_pickle=True)
        print(train_data[0].shape)
        print(train_data[0][0])
        train_label = np.load("data/train_label_{}.npy".format(dim),allow_pickle=True)
        #print(train_label[0].shape)
        #print(train_label[0][0])
        train_interval = np.load("data/train_interval_{}.npy".format(dim),allow_pickle=True)
        valid_data = np.load("data/valid_data_{}.npy".format(dim),allow_pickle=True)
        valid_label = np.load("data/valid_label_{}.npy".format(dim),allow_pickle=True)
        valid_interval = np.load("data/valid_interval_{}.npy".format(dim),allow_pickle=True)
        embedding = np.load("embedding.npy",allow_pickle=True)
        
        print(embedding.shape)

        

        if len(train_data[-1]) == 0 :
            train_data = train_data[:-1]
            train_label = train_label[:-1]
            train_interval = train_interval[:-1]
        if len(valid_data[-1]) == 0 :
            valid_data = valid_data[:-1]
            valid_label = valid_label[:-1]
            valid_interval = valid_interval[:-1]
        
        pos = 0
        total = 0
        
        mul_train = []
        for i,batch in enumerate(train_label):
            mul_train += [ batch.copy()]
            for j,label in enumerate(batch):
                pos += 1
                total += len(train_interval[i][j])-1
                for k in train_interval[i][j][1:]:
                    
                    mul_train[i][j][k-1] = 1
        
        mul_val = []
        for i,batch in enumerate(valid_label):
            mul_val += [ batch.copy()]
            for j,label in enumerate(batch):
                
                for k in valid_interval[i][j][1:]:
                    mul_val[i][j][k-1] = 1
        
         
        criterion =nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([(total-pos)/pos])).to(device)
        mymodel = SequenceTaggle(embedding.shape[0],embedding.shape[1],256,1,layer=3).to(device)
        mymodel.embedding.from_pretrained(torch.FloatTensor(embedding))
        solver.train(mymodel,train_data,train_label,mul_train,valid_data,valid_label,mul_val,criterion=criterion,device=device,epoch=int(arg[3]))
        if not os.path.exists("ckpt"):
            os.mkdir("ckpt")
        torch.save(mymodel.state_dict(), "ckpt/best.ckpt")
    else:
        #python main.py --test test_file pred_file TA/pred dim

        print("Testing")
        test_file = arg[2]
        dim = arg[5]
        dim = int(dim)
        if arg[4] == 'TA':
            dic = json.load(open("dict.json"))
            pre = Preprocessing()
            test_data, test_interval = pre.word_to_index(dic,test_file)
        else:
            if not os.path.isfile("embedding.npy") :
                pre = Preprocessing()
                pre.batch_data(dim=dim,modes=['train','valid','test'])

            test_data = np.load("data/test_data_{}.npy".format(dim),allow_pickle=True)
            test_interval = np.load("data/test_interval_{}.npy".format(dim),allow_pickle=True)
        embedding = np.load("embedding.npy",allow_pickle=True)
        if len(test_data[-1]) == 0 :
            test_data = test_data[:-1]
            test_interval = test_interval[:-1]
        print(test_data[0].shape[-1])
        mymodel = SequenceTaggle(embedding.shape[0],embedding.shape[1],256,1,layer=3).to(device)
        mymodel.embedding.from_pretrained(torch.FloatTensor(embedding))
        if os.path.isfile("ckpt/best.ckpt"):
            if torch.cuda.is_available():
                mymodel.load_state_dict(torch.load("ckpt/best.ckpt"))
            else:
                mymodel.load_state_dict(torch.load("ckpt/best.ckpt",map_location= device))
        elif os.path.isfile("best.ckpt"):
            if torch.cuda.is_available():
                mymodel.load_state_dict(torch.load("best.ckpt"))
            else:
                mymodel.load_state_dict(torch.load("best.ckpt",map_location= device))
        
        solver.test(mymodel,test_data,test_interval,arg[3],device=device,mode='test')

        
        