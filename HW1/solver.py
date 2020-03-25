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

    def train(self,seq_model,batches,labels,valid_batches,valid_labels,device,mode='extractive',
                criterion=nn.BCEWithLogitsLoss(),epoch=10,lr=0.0001,encoder=None,decoder=None):
        
        min_loss = 100000000
        best_model = None
        seq_opt=optim.RMSprop(seq_model.parameters(), lr=lr)
        if mode == 'extractive':
            for ep in range(epoch):
                seq_model.train()
                bl = len(batches)
                for i in range(bl):
                    seq_opt.zero_grad()
                    data = torch.LongTensor(batches[i]).to(device)
                    data = data.permute(1,0)
                    
                    target = torch.tensor(labels[i]).float().to(device)
                    #print(target.size())
                    target = target.permute(1,0)
                    
                    hidden = seq_model.encoder.initHidden(len(batches[i])).to(device)
                    pred, _ = seq_model(data,hidden)
                    

                    loss = criterion(pred.view(pred.size()[0],pred.size()[1])*target, target) 
                    loss.backward()

                    seq_opt.step()
                    

                    if i % 100 == 0:
                        #print(data)
                        print("Train epoch : {}, step : {} / {}, loss : {:.2f}".format(ep, i,bl,loss.item()))
                # validation
                seq_model.eval()
                bl = len(valid_batches)
                total_loss=0
                for i in range(bl):
                    data = torch.LongTensor(valid_batches[i]).to(device)
                    data = data.permute(1,0)
                    target = torch.tensor(valid_labels[i]).float().to(device)
                    target = target.permute(1,0)
                    
                    hidden = seq_model.encoder.initHidden(len(valid_batches[i])).to(device)
                    pred, _ = seq_model(data,hidden)
                    loss = criterion(pred.view(pred.size()[0],pred.size()[1])*target, target) 
                    total_loss += loss.item()
                    if i % 100 == 0:
                        print("Valid epoch : {}, step : {} / {}, loss : {}".format(ep, i,bl,loss.item()))
                if min_loss > total_loss:
                    min_loss =total_loss
                    best_model = seq_model
                if ep %10 == 0:
                    torch.save(best_model.state_dict(), "ckpt/best.ckpt")
            seq_model = best_model
                    

    def test(self,seq_model,batches,interval,output_file,device,mode='test'):
        result = []
        post = Postprocessing()
        n = 0
        result_dict = []
        l = len(batches)
        for i in range(l):
            
            data = torch.LongTensor(batches[i],device=device)
            data = data.permute(1,0)
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
        if mode == 'test':
            print('convert result to jsonl ...........')
            post.toJson(output_file,result_dict)













