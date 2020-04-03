from model import SequenceTaggle
import torch
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from postprocessing import Postprocessing
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
class Solver:
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def plot(self,x,y,x_val,y_val,epoch):
        plt.figure()
        plt.plot(x,y,"r",x_val,y_val,"b")
        plt.savefig("Epoch_{}.jpg".format(epoch))
    def train(self,seq_model,batches,valid_batches,device,attention=False,mode='abstractive',
                batch_size = 16,epoch=10,lr=0.0001,encoder=None,decoder=None):
        
        min_loss = 100000000
        best_model = None
        seq_opt=optim.RMSprop(seq_model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(seq_opt,step_size=1,gamma=0.5)
        step = 0
        x_train = []
        loss_train =[]
        x_val = []
        loss_val = []
        criterion = nn.CrossEntropyLoss().to(device)

        if mode == 'abstractive':
            t_bl = len(batches)
            
            v_bl = len(valid_batches)
            gap = 0.2
            for ep in range(epoch):
                seq_model.train()
                
                total_loss=0
                print(f'{ep} Start')

                for i,batch in enumerate(batches):
                    seq_opt.zero_grad()
                    teacher_force = True if random.random() < 0.5 else False
                    bs = batch['text'].size(0)
                    #print(bs)
                    data = batch['text'].to(device)
                    data = data.permute(1,0)
                    
                    target = batch['summary'].to(device)
                    
                    target = target.permute(1,0)
                    
                    #if not attention:
                    pred, _ ,_= seq_model(data,target[:-1])
                    pred = pred.permute(1,2,0)
                    target = target.permute(1,0)
                    #pred = pred.view(pred.size()[0],pred.size()[1])
                    loss = criterion(pred, target[:,1:]) 
                    """else:
                        
                        if teacher_force:
                            pred, hidden , att = seq_model(data,torch.LongTensor([1]*bs).view(1,-1).to(device))
                            loss = criterion(pred.permute(1,2,0), target.permute(1,0)[:,1].view(-1,1)) 
                            topv,topi = pred.topk(1)
                            #print(target[0:1].size())
                            #print(topi.view(1,-1).detach().size())
                            topi = topi.view(1,-1).detach()
                            for k in range(len(target[1:-1])):
                                
                                pred, hidden , att = seq_model.decoder(seq_model.embedding(target[k+1].view(1,-1)),hidden)
                                pred = seq_model.linear(pred)
                                loss += criterion(pred.permute(1,2,0), target.permute(1,0)[:,k+2].view(-1,1)) 
                                topv,topi = pred.topk(1)
                                topi = topi.view(1,-1).detach()
                            loss = loss/(len(target)-1)
                        else:
                        #bos
                            pred, hidden , att = seq_model(data,torch.LongTensor([1]*bs).view(1,-1).to(device))
                            loss = criterion(pred.permute(1,2,0), target.permute(1,0)[:,1].view(-1,1)) 
                            topv,topi = pred.topk(1)
                            #print(target[0:1].size())
                            #print(pred.size())
                            #print(topi.view(1,-1).detach().size())
                            topi = topi.view(1,-1).detach()
                            for k in range(len(target[1:-1])):
                                #print(target.permute(1,0)[:,i+1].view(-1,1))
                                pred, hidden , att = seq_model.decoder(seq_model.embedding(topi),hidden)
                                pred = seq_model.linear(pred)
                                #print(pred.permute(1,2,0).size(),target.permute(1,0)[:,i+1].view(-1,1).size())
                                loss += criterion(pred.permute(1,2,0), target.permute(1,0)[:,k+2].view(-1,1)) 
                                topv,topi = pred.topk(1)
                                topi = topi.view(1,-1).detach()
                            loss = loss/(len(target)-1) """
                            
                            
                    total_loss += loss.item() 
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(seq_model.parameters(),5)
                    seq_opt.step()
                    
                    step+=1
                    if i ==0:
                        x_train+= [step]
                        loss_train += [total_loss/(i+1)]
                            #print(pred.permute(1,0)[0])
    
                        print("Train epoch : {}, step : {} / {}, loss : {}".format(ep, i+1,t_bl,total_loss/(i+1)))
                    if (i+1) % 100 == 0:
                        #print(data)
                        x_train+= [step]
                        loss_train += [total_loss/(i+1)]
                            #print(pred.permute(1,0)[0])
    
                        print("Train epoch : {}, step : {} / {}, loss : {}".format(ep, i+1,t_bl,total_loss/(i+1)))
                
                # validation
                seq_model.eval()
               
                v_total_loss=0
                val_step = 0
            
                for i,batch in enumerate(valid_batches):
                    bs = batch['text'].size(0)
                    data = batch['text'].to(device)
                    data = data.permute(1,0)
                    
                    target = batch['summary'].to(device)
                    target = target.permute(1,0)
                    #if not attention:
                        
                        #print(target.size())
                    pred, _ ,_= seq_model(data,target[:-1])
                    pred = pred.permute(1,2,0)
                    target = target.permute(1,0)
                    #pred = pred.view(pred.size()[0],pred.size()[1])
                    loss = criterion(pred, target[:,1:]) 
                    """else:
                    #print(m.size())
                        pred, hidden , att = seq_model(data,torch.LongTensor([1]*bs).view(1,-1).to(device))
                        loss = criterion(pred.permute(1,2,0), target.permute(1,0)[:,1].view(-1,1)) 
                        topv,topi = pred.topk(1)
                        #print(target[0:1].size())
                        #print(pred.size())
                        #print(topi.view(1,-1).detach().size())
                        topi = topi.view(1,-1).detach()
                        for k in range(len(target[1:-1])):
                            #print(target.permute(1,0)[:,i+1].view(-1,1))
                            pred, hidden , att = seq_model.decoder(seq_model.embedding(topi),hidden)
                            pred = seq_model.linear(pred)
                            #print(pred.permute(1,2,0).size(),target.permute(1,0)[:,i+1].view(-1,1).size())
                            loss += criterion(pred.permute(1,2,0), target.permute(1,0)[:,k+2].view(-1,1)) 
                            topv,topi = pred.topk(1)
                            topi = topi.view(1,-1).detach()
                        loss = loss/(len(target)-1)"""
                    val_step+=1
                    v_total_loss+= loss.item()
                    if (i+1) % 100 == 0:
                        
                        print("Valid epoch : {}, step : {} / {}, loss : {}".format(ep,i+1,v_bl,v_total_loss/(i+1)))
                x_val+= [step]
                loss_val += [v_total_loss/v_bl]
                if min_loss > v_total_loss:
                    min_loss =v_total_loss
                    best_model = seq_model
                if  v_total_loss/v_bl - total_loss/t_bl > gap:
                    scheduler.step()
                    gap += 0.1
                if ep %5 == 0:
                    self.plot(x_train,loss_train,x_val,loss_val,epoch=ep)
                    torch.save(best_model.state_dict(), "ckpt/best.ckpt")
            self.plot(x_train,loss_train,x_val,loss_val,epoch=epoch)
            seq_model = best_model
                    
    def findEOS(self,sample):
        for i,word in enumerate(sample):
            if word == 2:
                return i
    def showAttention(self,input_sentence, output_words, attentions,tokenizer,p,s):
        # Set up figure with colorbar
        fig = plt.figure(figsize=(18,9))
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.detach().cpu().numpy(), cmap='bone')
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([""]+tokenizer.decode(input_sentence,not_ignore=True).split(" "), rotation=90)
        #print(output_words.detach().cpu().numpy())
        
        ax.set_yticklabels([""]+tokenizer.decode(output_words,not_ignore=True).split(" "))

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.savefig(f'{p}*128_{s}_att.png')
    def test(self,seq_model,batches,device,tokenizer,attention=False,batch_size=16,mode='test'):
        result = []
        post = Postprocessing()
        n = 0
        result_dict = []
        l = len(batches)
        seq_model.eval()
        max_len = 300
        if attention:
            for i in range(4):
                print(i)
                p_batch = [93//batch_size,96//batch_size,607//batch_size,649//batch_size]
                sen = [93%batch_size,97%batch_size,607%batch_size,649%batch_size]

                sample = batches[p_batch[i]]['text'][sen[i]].unsqueeze(1)
                stop = self.findEOS(sample)
                data = sample.to(device)
                pred, hidden , atts = seq_model(data,torch.LongTensor([1]).view(1,-1).to(device))
                
                topv,topi = pred.topk(1)    
                result = topi.view(1,-1).cpu().detach()
                
                for k in range(max_len-1):
                    
                    #print(result.size())
                    topi = topi.view(1,-1).detach()
                    #print(target.permute(1,0)[:,i+1].view(-1,1))
                    pred, hidden , att = seq_model.decoder(seq_model.embedding(topi),hidden)
                    atts = torch.cat((atts,att),dim=0)
                    pred = seq_model.linear(pred)
                    #print(pred.permute(1,2,0).size(),target.permute(1,0)[:,i+1].view(-1,1).size())
                    topv,topi = pred.topk(1)
                    
                    result = torch.cat((result,topi.cpu().view(1,-1).detach()),dim=0)
                    if topi.item() == 2:
                        break
                    #print(result.size())
                    """if topi.item == 2:
                        break"""
                    
                    #result += [topi.item()]
                print(result.size())
                atts = atts.squeeze(1)[:30,:stop]
                result = result.squeeze(1)

                #print(sample)
                #print(result)
                self.showAttention(sample,result,atts,tokenizer,p_batch[i],sen[i])
        for i,batch in enumerate(batches):
            
            bs = batch['text'].size(0)
            #print(bs)
            data = batch['text'].to(device)
            data = data.permute(1,0)
            pred, hidden , att = seq_model(data,torch.LongTensor([1]*bs).view(1,-1).to(device))
            topv,topi = pred.topk(1)
            #print(topi.view(1,-1))
            
            result = topi.view(1,-1)
            #print(target[0:1].size())
            #print(pred.size())
            #print(topi.view(1,-1).detach().size())
            
            
            for k in range(max_len-1):
                topi = topi.view(1,-1).detach()
                #print(target.permute(1,0)[:,i+1].view(-1,1))
                pred, hidden , att = seq_model.decoder(seq_model.embedding(topi),hidden)
                pred = seq_model.linear(pred)
                #print(pred.permute(1,2,0).size(),target.permute(1,0)[:,i+1].view(-1,1).size())
                topv,topi = pred.topk(1)
                result = torch.cat((result,topi.view(1,-1)),dim=0)
                #print(result.size())
                """if topi.item == 2:
                    break"""
                
                #result += [topi.item()]
            result_dict += [result.permute(1,0).cpu().numpy()]
                
                
            if (i+1) %10 == 0:
                print(f"{i+1}/{l}")
            
            
            #print(pred.size())
        
        return result_dict













