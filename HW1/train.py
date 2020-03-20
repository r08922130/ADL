from model import SequenceTaggle
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from postprocessing import Postprocessing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(seq_model,batches,labels,device=device,mode='extractive',
            criterion=nn.BCEWithLogitsLoss(),epoch=10,encoder=None,decoder=None):
    total_loss=0
    seq_opt=optim.RMSprop(seq_model.parameters(), lr=1)
    if mode == 'extractive':
        for ep in range(epoch):
            for i in range(len(batches)):
                seq_opt.zero_grad()
                data = torch.tensor(batches[i],device=device).float()
                data = data.permute(1,0,2)
                target = torch.tensor(labels[i],device=device).float()
                #print(target.size())
                target = target.permute(1,0)
                
                pred, _ = seq_model(data)
                loss = criterion(pred.view(pred.size()[0],pred.size()[1]), target) 
                loss.backward()

                seq_opt.step()
                total_loss += loss.item()

                print(loss)
                if i == 50:
                    break
def test(seq_model,batches,interval,mode='test',device=device):
    result = []
    post = Postprocessing()
    n = 0
    result_dict = []
    for i in range(len(batches)):
        
        data = torch.tensor(batches[i],device=device).float()
        data = data.permute(1,0,2)
        pred, _ = seq_model(data)
        pred = pred.view(pred.size()[0],pred.size()[1])
        pred = pred.permute(1,0)
        if i == 0:
            print(pred)
        pred = pred > 0.5
        pred = pred.float()
        #print(pred.size())
        result_dict,n = post.select_sentence(pred.numpy(),interval[i],result_dict,n)
        
        #print(pred.size())
    print('convert result to jsonl ...........')
    post.toJson('early.jsonl',result_dict)

mymodel = SequenceTaggle(53,256,1,device)



train_data = np.load("data/train_data.npy",allow_pickle=True)
train_label = np.load("data/train_label.npy",allow_pickle=True)
train_interval = np.load("data/train_interval.npy",allow_pickle=True)

test_data = np.load("data/test_data.npy",allow_pickle=True)
test_interval = np.load("data/test_interval.npy",allow_pickle=True)


if len(train_data[-1]) == 0 :
    train_data = train_data[:-1]
    train_label = train_label[:-1]
    train_interval = train_interval[:-1]
if len(test_data[-1]) == 0 :
    test_data = test_data[:-1]
    test_interval = test_interval[:-1]
print(train_data.shape)
print(train_label.shape)
print(train_interval.shape)
pos = 0
total = 0
for i,batch in enumerate(train_label):
    for j,label in enumerate(batch):
        pos += sum(label[:train_interval[i][j][-1]])
        total += len(label[:train_interval[i][j][-1]])
        #print(pos,total)
criterion =nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([(total-pos)/pos]))
train(mymodel,train_data,train_label,criterion=criterion,device=device,epoch=1)

test(mymodel,test_data,test_interval,mode='test')


