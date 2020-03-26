import torch.nn as nn
import torch.nn.functional as F
import torch
class SequenceTaggle(nn.Module):
    def __init__(self,num_embeddings, embedding_dim,hidden_size,output_size,device,layer=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.sen_emb = SentenceEncoder(embedding_dim,hidden_size,device,layer=layer)
        self.encoder = Encoder(hidden_size,hidden_size,device,layer=layer)
        self.linear = nn.Linear(hidden_size*2,output_size)
        #self.linear1 = nn.Linear(hidden_size,output_size)
        self.layer = layer
        
    def forward(self,input):
        store = torch.tensor([])
        #store = torch.ones((1,input.size(-1),self.hidden_size))
        for sentence in input: 
            output = self.embedding(sentence)
            hidden = self.sen_emb.initHidden(input.size(-1),self.layer*2)

            cell, hidden = self.sen_emb(output,hidden)
            #print(hidden.size())
            #cell = self.linear1(cell)
            cell = torch.mean(cell,dim=0)
            cell = cell.view(1,cell.size(0),cell.size(1))
            store = torch.cat((store,cell),dim=0)
            
        print(store.size())
        output,hidden = self.encoder(store)
        #print(output.size())
        output = self.linear(output)
        output = torch.sigmoid(output)
        return output,hidden

class SentenceEncoder(nn.Module):
    def __init__(self,input_size,hidden_size,device,layer=1,batch_size=16):
        super().__init__()
        self.input_size = input_size
        self.device = device
        self.encoder = Encoder(input_size,hidden_size,device,layer=layer)
        self.linear = nn.Linear(hidden_size*2,hidden_size)
        self.hidden_size = hidden_size
    def forward(self,input,hidden):
        # one sentence of each data in a batch 
        output , hidden = self.encoder(input)
        output = self.linear(output)
        output = torch.tanh(output)
        return output , hidden
    def initHidden(self,batch,layer):
        return torch.zeros(layer,batch,self.hidden_size).to(self.device)


class Encoder(nn.Module):
    def __init__(self,input_size,hidden_size,device,layer=1,batch_size=16):
        super().__init__()
        self.batch_size = batch_size
        self.layer = layer
        self.hidden_size = hidden_size
        self.device =device
        self.linear = nn.Linear(input_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size,num_layers= layer, bidirectional=True)
        self.LN = nn.LayerNorm(hidden_size*2)
        self.gru_F = nn.GRU(input_size,hidden_size,num_layers= layer)
        self.LN_F = nn.LayerNorm(hidden_size)
    def forward(self,input):
        
        hidden = self.initHidden(input.size(1),self.layer)
        output , hidden =self.gru_F(input,hidden)
        output = self.LN_F(output)
        hidden = self.initHidden(input.size(1),self.layer*2)
        output , hidden =self.gru(output,hidden)
        
        output = self.LN(output)
        return output , hidden
    def initHidden(self,batch,layer):
        return torch.zeros(layer,batch,self.hidden_size).to(self.device)
