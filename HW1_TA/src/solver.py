from model import SequenceTaggle
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from postprocessing import Postprocessing
class Solver:
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def train(self,seq_model,batches,labels,valid_batches,valid_labels,device,mode='extractive',
                batch_size = 16,epoch=10,lr=0.00001,encoder=None,decoder=None):
        
        min_loss = 100000000
        best_model = None
        seq_opt=optim.RMSprop(seq_model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(seq_opt,step_size=10,gamma=0.5)
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
                bl = bl//batch_size+1
                for i in range(bl):
                    seq_opt.zero_grad()
                    data = batches[i*batch_size:(i+1)*batch_size].to(device)
                    data = data.permute(1,0)
                    
                    target = labels[i*batch_size:(i+1)*batch_size].float().to(device)
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
                    if i % 100 == 0:
                        #print(data)
                        x_train+= [step]
                        loss_train += [total_loss/(i+1)]
                            #print(pred.permute(1,0)[0])
                        print("Train epoch : {}, step : {} / {}, loss : {}".format(ep, i,bl,loss.item()))
                scheduler.step()
                # validation
                seq_model.eval()
                bl = len(valid_batches)
                bl = bl//batch_size +1
                total_loss=0
                val_step = 0
                criterion =nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1])).to(device)
            
                for i in range(bl):
                    data = valid_batches[i*batch_size:(i+1)*batch_size].to(device)
                    data = data.permute(1,0)
                    target = valid_labels[i*batch_size:(i+1)*batch_size].float().to(device)
                    target = target.permute(1,0)
                    
                    
                    pred, _ = seq_model(data)
                    loss = criterion(pred.view(pred.size()[0],pred.size()[1]), target) 

                    total_loss += loss.item()
                    val_step+=1
                    
                    if i % 100 == 0:
                        
                        print("Valid epoch : {}, step : {} / {}, loss : {}".format(ep, i,bl,loss.item()))
                x_val+= [step]
                loss_val += [total_loss/bl]
                if min_loss > total_loss:
                    min_loss =total_loss
                    best_model = seq_model
                if ep %10 == 0:
                    torch.save(best_model.state_dict(), "ckpt/best.ckpt")
            
            seq_model = best_model
                    
    
    def test(self,seq_model,batches,interval,output_file,device,batch_size=16,mode='test',model='m1',threshold=0.4):
        result = []
        post = Postprocessing()
        n = 0
        result_dict = []
        l = len(batches)
        l = l //batch_size+1
        for i in range(l):
            
            data = torch.LongTensor(batches[i*batch_size:(i+1)*batch_size]).to(device)
            
            data = data.permute(1,0)
            
            pred, _ = seq_model(data)
            pred = pred.view(pred.size()[0],pred.size()[1])
            pred = pred.permute(1,0)
            pred = torch.sigmoid(pred)
            if i %500 == 0:
                print(i/l)
            
            pred = pred > threshold
            
            pred = pred.detach().float()
            #print(pred.size())
            result_dict,n = post.select_sentence(pred.cpu().numpy(),interval[i],result_dict,n,model=model)
            
            #print(pred.size())
        if mode == 'test':
            print('convert result to jsonl ...........')
            post.toJson(output_file,result_dict)













