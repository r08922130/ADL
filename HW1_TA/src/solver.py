from model import SequenceTaggle
import torch
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from postprocessing import Postprocessing
import matplotlib.pyplot as plt

class Solver:
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def plot(self,x,y,x_val,y_val,epoch):
        plt.figure()
        plt.plot(x,y,"r",x_val,y_val,"b")
        plt.savefig("Epoch_{}.jpg".format(epoch))
    def train(self,seq_model,batches,valid_batches,device,attention=False,mode='abstractive',
                batch_size = 16,epoch=10,lr=0.00001,encoder=None,decoder=None):
        
        min_loss = 100000000
        best_model = None
        seq_opt=optim.RMSprop(seq_model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(seq_opt,step_size=2,gamma=0.9)
        step = 0
        x_train = []
        loss_train =[]
        x_val = []
        loss_val = []
        criterion = nn.CrossEntropyLoss().to(device)

        if mode == 'abstractive':
            t_bl = len(batches)
            
            v_bl = len(valid_batches)
            
            for ep in range(epoch):
                seq_model.train()
                
                total_loss=0
                print(f'{ep} Start')

                for i,batch in enumerate(batches):
                    seq_opt.zero_grad()
                    teacher_force = True if random.random() < 0.5 else False
                    
                    data = batch['text'].to(device)
                    data = data.permute(1,0)
                    
                    target = batch['summary'].to(device)
                    
                    target = target.permute(1,0)
                    
                    if not attention:
                        pred, _ ,_= seq_model(data,target[:-1])
                        pred = pred.permute(1,2,0)
                        target = target.permute(1,0)
                        #pred = pred.view(pred.size()[0],pred.size()[1])
                        loss = criterion(pred, target[:,1:]) 
                    else:
                        
                        if teacher_force:
                            pred, hidden , att = seq_model(data,torch.LongTensor([1]*batch_size).view(1,-1).to(device))
                            loss = criterion(pred.permute(1,2,0), target.permute(1,0)[:,i+1].view(-1,1)) 
                            topv,topi = pred.topk(1)
                            #print(target[0:1].size())
                            #print(topi.view(1,-1).detach().size())
                            topi = topi.view(1,-1).detach()
                            for i in range(len(target[1:-1])):
                                #print(target.permute(1,0)[:,i+1].view(-1,1))
                                pred, hidden , att = seq_model.decoder(seq_model.embedding(topi),hidden)
                                pred = seq_model.linear(pred)
                                loss += criterion(pred.permute(1,2,0), target.permute(1,0)[:,i+1].view(-1,1)) 
                                topv,topi = pred.topk(1)
                                topi = topi.view(1,-1).detach()
                            
                        else:
                        #bos
                            pred, hidden , att = seq_model(data,torch.LongTensor([1]*batch_size).view(1,-1).to(device))
                            loss = criterion(pred.permute(1,2,0), target.permute(1,0)[:,i+1].view(-1,1)) 
                            topv,topi = pred.topk(1)
                            #print(target[0:1].size())
                            #print(pred.size())
                            #print(topi.view(1,-1).detach().size())
                            topi = topi.view(1,-1).detach()
                            for i in range(len(target[1:-1])):
                                #print(target.permute(1,0)[:,i+1].view(-1,1))
                                pred, hidden , att = seq_model.decoder(seq_model.embedding(topi),hidden)
                                pred = seq_model.linear(pred)
                                #print(pred.permute(1,2,0).size(),target.permute(1,0)[:,i+1].view(-1,1).size())
                                loss += criterion(pred.permute(1,2,0), target.permute(1,0)[:,i+1].view(-1,1)) 
                                topv,topi = pred.topk(1)
                                topi = topi.view(1,-1).detach()
                                
                            
                            
                    total_loss += loss.item() 
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(seq_model.parameters(),5)
                    
                    
                    step+=1
                    total_loss+= loss.item()
                    if i % 100 == 0:
                        #print(data)
                        x_train+= [step]
                        loss_train += [total_loss/(i+1)]
                            #print(pred.permute(1,0)[0])
    
                        print("Train epoch : {}, step : {} / {}, loss : {}".format(ep, i,t_bl,loss.item()))
                scheduler.step()
                # validation
                seq_model.eval()
               
                total_loss=0
                val_step = 0
            
                for i,batch in enumerate(valid_batches):
                    
                    data = batch['text'].to(device)
                    data = data.permute(1,0)
                    
                    target = batch['summary'].to(device)
                    target = target.permute(1,0)
                    if not attention:
                        
                        #print(target.size())
                        pred, _ ,_= seq_model(data,target[:-1])
                        pred = pred.permute(1,2,0)
                        target = target.permute(1,0)
                        #pred = pred.view(pred.size()[0],pred.size()[1])
                        loss = criterion(pred, target[:,1:]) 
                    else:
                    #print(m.size())
                        pred, hidden , att = seq_model(data,torch.LongTensor([1]*batch_size).view(1,-1).to(device))
                        loss = criterion(pred.permute(1,2,0), target.permute(1,0)[:,i+1].view(-1,1)) 
                        topv,topi = pred.topk(1)
                        #print(target[0:1].size())
                        #print(pred.size())
                        #print(topi.view(1,-1).detach().size())
                        topi = topi.view(1,-1).detach()
                        for i in range(len(target[1:-1])):
                            #print(target.permute(1,0)[:,i+1].view(-1,1))
                            pred, hidden , att = seq_model.decoder(seq_model.embedding(topi),hidden)
                            pred = seq_model.linear(pred)
                            #print(pred.permute(1,2,0).size(),target.permute(1,0)[:,i+1].view(-1,1).size())
                            loss += criterion(pred.permute(1,2,0), target.permute(1,0)[:,i+1].view(-1,1)) 
                            topv,topi = pred.topk(1)
                            topi = topi.view(1,-1).detach()
                    
                    val_step+=1
                    
                    if i % 100 == 0:
                        
                        print("Valid epoch : {}, step : {} / {}, loss : {}".format(ep, i,v_bl,loss.item()))
                x_val+= [step]
                loss_val += [total_loss/v_bl]
                if min_loss > total_loss:
                    min_loss =total_loss
                    best_model = seq_model
                else:
                    seq_opt.step()
                if ep %5 == 0:
                    self.plot(x_train,loss_train,x_val,loss_val,epoch=ep)
                    torch.save(best_model.state_dict(), "ckpt/best.ckpt")
            self.plot(x_train,loss_train,x_val,loss_val,epoch=epoch)
            seq_model = best_model
                    
    
    def test(self,seq_model,batches,interval,output_file,device,batch_size=16,mode='test',model='m1',threshold=0.4):
        result = []
        post = Postprocessing()
        n = 0
        result_dict = []
        l = len(batches)
        for i in range(l):
            result=[]
            pred, hidden , att = seq_model(data,torch.LongTensor([1]*batch_size).view(1,-1).to(device))
            topv,topi = pred.topk(1)
            result += [topi.item()]
            #print(target[0:1].size())
            #print(pred.size())
            #print(topi.view(1,-1).detach().size())
            
            
            for i in range(len(target[1:-1])):
                topi = topi.view(1,-1).detach()
                #print(target.permute(1,0)[:,i+1].view(-1,1))
                pred, hidden , att = seq_model.decoder(seq_model.embedding(topi),hidden)
                pred = seq_model.linear(pred)
                #print(pred.permute(1,2,0).size(),target.permute(1,0)[:,i+1].view(-1,1).size())
                topv,topi = pred.topk(1)
                if topi.item == 2:
                    break
                result += [topi.item()]

                
                
            if i %500 == 0:
                print(i/l)
            result_dict += [result]
            
            
            #print(pred.size())
        
        return result_dict













