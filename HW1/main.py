import numpy as np
import torch
import torch.nn as nn 
import sys
from preprocessing import Preprocessing
from model import SequenceTaggle
import os
from solver import Solver
if __name__ == "__main__":
    
    arg = sys.argv
    solver = Solver()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if arg[1] == '--train':
        # cmd : python main.py --train emb_dim 30 
        dim = arg[2] if arg[2] else 50
        dim = int(dim)
        if not os.path.isfile("data/train_data_{}.npy".format(dim)):
            pre = Preprocessing("data/train.jsonl")
            train_data, train_label,train_interval = pre.batch_data(dim = dim,mode='train')

            np.save("data/train_data_{}.npy".format(dim),train_data)
            np.save("data/train_label_{}.npy".format(dim),train_label)
            np.save("data/train_interval_{}.npy".format(dim),train_interval)
        else:
            train_data = np.load("data/train_data_{}.npy".format(dim),allow_pickle=True)
            train_label = np.load("data/train_label_{}.npy".format(dim),allow_pickle=True)
            train_interval = np.load("data/train_interval_{}.npy".format(dim),allow_pickle=True)
        if not os.path.isfile("data/valid_data_{}.npy".format(dim)):
            pre = Preprocessing("data/valid.jsonl")
            valid_data, valid_label,valid_interval = pre.batch_data(dim=dim,mode='valid')

            np.save("data/valid_data_{}.npy".format(dim),valid_data)
            np.save("data/valid_label_{}.npy".format(dim),valid_label)
            np.save("data/valid_interval_{}.npy".format(dim),valid_interval)
        else:
            valid_data = np.load("data/valid_data_{}.npy".format(dim),allow_pickle=True)
            valid_label = np.load("data/valid_label_{}.npy".format(dim),allow_pickle=True)
            valid_interval = np.load("data/valid_interval_{}.npy".format(dim),allow_pickle=True)

        

        

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
        for i,batch in enumerate(train_label):
            for j,label in enumerate(batch):
                pos += sum(label[:train_interval[i][j][-1]])
                total += len(label[:train_interval[i][j][-1]])
        criterion =nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([(total-pos)/pos])).to(device)
        mymodel = SequenceTaggle(train_data[0].shape[-1],256,1,device=device,layer=3).to(device)
        solver.train(mymodel,train_data,train_label,valid_data,valid_label,valid_interval,criterion=criterion,device=device,epoch=int(arg[3]))
        if not os.path.exists("ckpt"):
            os.mkdir("ckpt")
        torch.save(mymodel.state_dict(), "ckpt/best.ckpt")
    else:
        #python main.py --test test_file pred_file TA/pred dim
        test_file = arg[2]
        dim = arg[5]
        dim = int(dim)
        if arg[4] == 'TA':
            pre = Preprocessing(test_file)

            test_data, _,test_interval = pre.batch_data(dim=dim,mode='test')
        else:    
            if not os.path.isfile("data/test_data_{}.npy".format(dim)) :
                pre = Preprocessing(test_file)

                test_data, _,test_interval = pre.batch_data(dim=dim,mode='test')
                np.save("data/test_data_{}.npy".format(dim),test_data)
                np.save("data/test_interval_{}.npy".format(dim),test_interval)
            else:
                test_data = np.load("data/test_data_{}.npy".format(dim),allow_pickle=True)
                test_interval = np.load("data/test_interval_{}.npy".format(dim),allow_pickle=True)
        if len(test_data[-1]) == 0 :
            test_data = test_data[:-1]
            test_interval = test_interval[:-1]
        print(test_data[0].shape[-1])
        mymodel = SequenceTaggle(test_data[0].shape[-1],256,1,device=device,layer=3).to(device)
        if os.path.isfile("ckpt/best.ckpt"):
            mymodel.load_state_dict(torch.load("ckpt/best.ckpt"))
        solver.test(mymodel,test_data,test_interval,arg[3],device=device,mode='test')

        
        