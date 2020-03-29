import numpy as np
import torch
import torch.nn as nn 
import sys
from preprocessing import Preprocessing
from model import SequenceTaggle
from model import SequenceTaggle1
import os
from solver import Solver
import json
if __name__ == "__main__":
    
    arg = sys.argv
    solver = Solver()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if arg[1] == '--train':
        print("Training")
        # cmd : python main.py --train emb_dim 30 pre/no m1/m2
        #                0        1       2     3    4    5
        dim = arg[2] if arg[2] else 50
        dim = int(dim)
        if not os.path.isfile("data/train_data_{}_{}.npy".format(dim,arg[5])):
            print("No File . Preprocessing!")
            pre = Preprocessing()
            dic = json.load(open("dict.json"))
            if arg[5] == 'm1':
                pre.batch_data(dic,dim=dim,modes=['train','valid','test'],model=arg[5])
            else:
                pre.pad_sentences(dic,dim=dim,modes=['train','valid','test'],model=arg[5])
        
        train_data = np.load("data/train_data_{}_{}.npy".format(dim,arg[5]),allow_pickle=True)
        print(train_data[0].shape)
        print(train_data[0][0])
        train_label = np.load("data/train_label_{}_{}.npy".format(dim,arg[5]),allow_pickle=True)
        #print(train_label[0].shape)
        #print(train_label[0][0])
        train_interval = np.load("data/train_interval_{}_{}.npy".format(dim,arg[5]),allow_pickle=True)
        valid_data = np.load("data/valid_data_{}_{}.npy".format(dim,arg[5]),allow_pickle=True)
        valid_label = np.load("data/valid_label_{}_{}.npy".format(dim,arg[5]),allow_pickle=True)
        valid_interval = np.load("data/valid_interval_{}_{}.npy".format(dim,arg[5]),allow_pickle=True)
        embedding = np.load("embedding.npy",allow_pickle=True)
        
        print(embedding.shape)

        # data shape :(batch,# sentences per data, # word per sentences)

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
        if arg[5] == 'm1':
            
            for i,batch in enumerate(train_label):
                for j,label in enumerate(batch):
                    pos += np.sum(label)
                    total += label.shape[0]
                    
            
            
            
            print(pos)
            print(total)
            criterion =nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([(total-pos)/pos/2])).to(device)
            mymodel = SequenceTaggle1(embedding.shape[0],embedding.shape[1],256,1,device,layer=4).to(device)
            mymodel.embedding.from_pretrained(torch.FloatTensor(embedding))
            if arg[4] == 'pre':
                mymodel.load_state_dict(torch.load("ckpt/best.ckpt"))
            solver.train(mymodel,train_data,train_label,valid_data,valid_label,criterion=criterion,device=device,epoch=int(arg[3]))
        
        else:
            for i,batch in enumerate(train_label):
                
                for j,label in enumerate(batch):
                    pos += 1
                    total += len(batch)
            print(pos)
            print(total)
            mymodel = SequenceTaggle(embedding.shape[0],embedding.shape[1],256,1,device,layer=3).to(device)
            mymodel.embedding.from_pretrained(torch.FloatTensor(embedding))
            if arg[4] == 'pre':
                mymodel.load_state_dict(torch.load("ckpt/best.ckpt"))
            solver.train_sentences(mymodel,train_data,train_label,valid_data,valid_label,device=device,epoch=int(arg[3]))
        if not os.path.exists("ckpt"):
            os.mkdir("ckpt")
        
        torch.save(mymodel.state_dict(), "ckpt/best.ckpt")
    else:
        #python main.py --test test_file pred_file TA/pred dim  m1/m2 ckpt
        #         0      1        2          3         4     5  6      7
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

            test_data = np.load("data/{}_data_{}_{}.npy".format(test_file,dim,arg[6]),allow_pickle=True)
            test_interval = np.load("data/{}_interval_{}_{}.npy".format(test_file,dim,arg[6]),allow_pickle=True)
        embedding = np.load("embedding.npy",allow_pickle=True)
        if len(test_data[-1]) == 0 :
            test_data = test_data[:-1]
            test_interval = test_interval[:-1]
        print(test_data[0].shape[-1])
        if arg[6] == 'm1' :
            mymodel = SequenceTaggle1(embedding.shape[0],embedding.shape[1],256,1,device,layer=4).to(device)
            mymodel.embedding.from_pretrained(torch.FloatTensor(embedding))
        
        else:
            mymodel = SequenceTaggle(embedding.shape[0],embedding.shape[1],256,1,device,layer=3).to(device)
            mymodel.embedding.from_pretrained(torch.FloatTensor(embedding))
        if os.path.isfile(arg[7]):
            if torch.cuda.is_available():
                mymodel.load_state_dict(torch.load(arg[7]))
            else:
                mymodel.load_state_dict(torch.load(arg[7],map_location= device))
        
        solver.test(mymodel,test_data,test_interval,arg[3],device=device,mode='test',model=arg[6])

        
        