from model import SequenceTaggle
import torch
import numpy as np
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
    def train(self,seq_model,batches,labels,valid_batches,valid_labels,device,mode='extractive',
                criterion=nn.BCEWithLogitsLoss(),epoch=10,lr=0.00001,encoder=None,decoder=None):
        
        min_loss = 100000000
        best_model = None
        seq_opt=optim.RMSprop(seq_model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(seq_opt,step_size=2,gamma=0.85)
        step = 0
        x_train = []
        loss_train =[]
        x_val = []
        loss_val = []
        if mode == 'extractive':
            for ep in range(epoch):
                seq_model.train()
                bl = len(batches)
                total_loss=0
                
                for i in range(bl):
                    seq_opt.zero_grad()
                    data = torch.LongTensor(batches[i]).to(device)
                    data = data.permute(1,0)
                    
                    target = torch.tensor(labels[i]).float().to(device)
                    #print(target.size())
                    target = target.permute(1,0)
                    pos = torch.sum(target)
                    total =  target.size(0) * target.size(1)
                    criterion =nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([(total-pos)/pos])).to(device)

                    #print(m.size())
                    
                    pred, _ = seq_model(data)
                    
                    pred = pred.view(pred.size()[0],pred.size()[1])
                    loss = criterion(pred, target) 

                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(seq_model.parameters(),5)
                    seq_opt.step()
                    
                    step+=1
                    total_loss+= loss.item()
                    

                    if i == 0 or (i+1) % 100 == 0:
                        #print(data)
                        x_train+= [step]
                        loss_train += [total_loss/(i+1)]
                            #print(pred.permute(1,0)[0])
                        print("Train epoch : {}, step : {} / {}, loss : {}".format(ep, i+1,bl,total_loss/(i+1)))
                
                # validation
                seq_model.eval()
                bl = len(valid_batches)
                total_loss=0
                val_step = 0
                criterion =nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1])).to(device)
                for i in range(bl):
                    data = torch.LongTensor(valid_batches[i]).to(device)
                    data = data.permute(1,0)
                    target = torch.tensor(valid_labels[i]).float().to(device)
                    target = target.permute(1,0)
                    
                    
                    pred, _ = seq_model(data)
                    loss = criterion(pred.view(pred.size()[0],pred.size()[1]), target) 

                    total_loss += loss.item()
                    val_step+=1
                    
                    if (i+1) % 100 == 0:
                        
                        print("Valid epoch : {}, step : {} / {}, loss : {}".format(ep, i+1,bl,total_loss/(i+1)))
                x_val+= [step]
                loss_val += [total_loss/bl]
                if min_loss > total_loss:
                    min_loss =total_loss
                    #best_model = 
                    torch.save(seq_model.state_dict(), "ckpt/best.ckpt")
                else:
                    scheduler.step()
                if ep %5 == 0:
                    self.plot(x_train,loss_train,x_val,loss_val,epoch)
                    
            self.plot(x_train,loss_train,x_val,loss_val,epoch)
            #seq_model = best_model
                    
    def train_sentences(self,seq_model,batches,labels,valid_batches,valid_labels,device,mode='extractive',
                criterion=nn.BCEWithLogitsLoss(),epoch=10,lr=0.0001,encoder=None,decoder=None):
        
        min_loss = 100000000
        best_model = None
        seq_opt=optim.Adam(seq_model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(seq_opt,step_size=2,gamma=0.85)
        step = 0
        x_train = []
        loss_train =[]
        x_val = []
        loss_val = []
        if mode == 'extractive':
            for ep in range(epoch):
                seq_model.train()
                bl = len(batches)
                total_loss = 0
                
                for i in range(bl):
                    seq_opt.zero_grad()
                    data = torch.LongTensor(batches[i]).to(device)
                    data = data.permute(1,2,0)
                    
                    target = torch.tensor(labels[i]).float().to(device)
                    #print(target.size())
                    target = target.permute(1,0)
                    
                    
                    
                    pred, _ = seq_model(data)
                    
                    pred = pred.view(pred.size()[0],pred.size()[1])
                    loss = criterion(pred, target) 

                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(seq_model.parameters(),1)
                    seq_opt.step()
                    

                    step+=1
                    total_loss+= loss.item()
                    if i == 0 or (i+1) % 100 == 0:
                        #print(data)
                        x_train+= [step]
                        loss_train += [total_loss/(i+1)]
                            #print(pred.permute(1,0)[0])
                        print("Train epoch : {}, step : {} / {}, loss : {}".format(ep, i+1,bl,total_loss/(i+1)))
                
                # validation
                seq_model.eval()
                bl = len(valid_batches)
                total_loss=0
                val_step = 0
                for i in range(bl):
                    data = torch.LongTensor(valid_batches[i]).to(device)
                    data = data.permute(1,2,0)
                    
                    target = torch.tensor(valid_labels[i]).float().to(device)
                    target = target.permute(1,0)
                    
                    
                    pred, _ = seq_model(data)
                    loss = criterion(pred.view(pred.size()[0],pred.size()[1]), target) 

                    total_loss += loss.item()
                    val_step +=1
                    if (i+1) % 100 == 0:
                        
                        print("Valid epoch : {}, step : {} / {}, loss : {}".format(ep, i+1,bl,total_loss/(i+1)))
                x_val+= [step]
                loss_val += [total_loss/bl]
                if min_loss > total_loss:
                    min_loss =total_loss
                    #best_model = seq_model
                    torch.save(seq_model.state_dict(), "ckpt/best.ckpt")
                else:
                    scheduler.step()
                if ep %5 == 0:
                    self.plot(x_train,loss_train,x_val,loss_val,epoch)
                    #torch.save(best_model.state_dict(), "ckpt/best.ckpt")
            self.plot(x_train,loss_train,x_val,loss_val)
            #torch.save(best_model.state_dict(), "ckpt/best.ckpt")
            seq_model = best_model
    def test(self,seq_model,batches,interval,b_ids,output_file,device,mode='test',model='m1',threshold=0.8):
        result = []
        post = Postprocessing()
        n = 0
        result_dict = []
        result_hist = []
        l = len(batches)
        for i in range(l):
            ids = b_ids[i]
            data = torch.LongTensor(batches[i]).to(device)
            if model == 'm1':
                data = data.permute(1,0)
            else:
                data = data.permute(1,2,0)
            pred, _ = seq_model(data)
            pred = pred.view(pred.size()[0],pred.size()[1])
            pred = pred.permute(1,0)
            pred = torch.sigmoid(pred)
            
            if (i+1) %100 == 0:
                print(f"{i+1}/{l}")
            if model == 'm1':
                pred = pred > threshold
            

            pred = pred.detach().float()
            #if i == 0 :
                #print(pred[2])
                #print(pred[3])
            #print(pred.size())
            result_dict,result_hist,n = post.select_sentence2(pred.cpu().numpy(),interval[i],ids,result_dict,result_hist,n,mode=mode,model=model)
            
            #print(pred.size())
        # show relative location
        # hist shape (# batches, batch size, predicts)
        num_of_bins = 25
        plt.figure()
        plt.hist(result_hist,bins=num_of_bins,range=(0,1))
        plt.savefig("extractive.png")

        
        print('convert result to jsonl ...........')
        post.toJson(output_file,result_dict)













