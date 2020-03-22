from model import SequenceTaggle
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from postprocessing import Postprocessing
class Solver:
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self,seq_model,batches,labels,valid_batches,valid_labels,valid_intervals,device,mode='extractive',
                criterion=nn.BCEWithLogitsLoss(),epoch=10,lr=0.0001,encoder=None,decoder=None):
        total_loss=0
        seq_opt=optim.RMSprop(seq_model.parameters(), lr=lr)
        if mode == 'extractive':
            for ep in range(epoch):
                
                for i in range(len(batches)):
                    seq_opt.zero_grad()
                    data = torch.tensor(batches[i],device=device).float()
                    data = data.permute(1,0,2)
                    
                    target = torch.tensor(labels[i],device=device).float()
                    #print(target.size())
                    target = target.permute(1,0)
                    hidden = seq_model.encoder.initHidden(len(batches[i]))
                    pred, _ = seq_model(data,hidden)
                    loss = criterion(pred.view(pred.size()[0],pred.size()[1]), target) 
                    loss.backward()

                    seq_opt.step()
                    total_loss += loss.item()
                    if i % 100 == 0:
                        print(loss)
                    
    def test(self,seq_model,batches,interval,output_file,device,mode='test'):
        result = []
        post = Postprocessing()
        n = 0
        result_dict = []
        l = len(batches)
        for i in range(l):
            
            data = torch.tensor(batches[i],device=device).float()
            data = data.permute(1,0,2)
            hidden = seq_model.encoder.initHidden(len(batches[i]))
            pred, _ = seq_model(data,hidden)
            pred = pred.view(pred.size()[0],pred.size()[1])
            pred = pred.permute(1,0)
            if i %500 == 0:
                print(i/l)
            pred = pred > 0.5
            pred = pred.float()
            #print(pred.size())
            result_dict,n = post.select_sentence(pred.cpu().numpy(),interval[i],result_dict,n)
            
            #print(pred.size())
        print('convert result to jsonl ...........')
        post.toJson(output_file,result_dict)













