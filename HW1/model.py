import torch.nn as nn
import torch.nn.functional as F
import torch
class SequenceTaggle(nn.Module):
    def __init__(self,num_embeddings, embedding_dim,hidden_size,output_size,device,layer=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        self.encoder = Encoder(embedding_dim,hidden_size,device,layer=layer)
        self.linear = nn.Linear(hidden_size*2,output_size)
        
        
    def forward(self,input):
        output = self.embedding(input)
        output,hidden = self.encoder(output)
        output = self.linear(output)
        output = torch.sigmoid(output)
        return output,hidden

        

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
