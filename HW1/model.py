import torch.nn as nn
import torch.nn.functional as F
import torch
class SequenceTaggle(nn.Module):
    def __init__(self,num_embeddings, embedding_dim,hidden_size,output_size,layer=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        self.encoder = Encoder(embedding_dim,hidden_size,layer)
        self.linear = nn.Linear(hidden_size*2,output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,input,hidden,m):
        output = self.embedding(input)
        output,hidden = self.encoder(output,hidden)
        output = self.linear(output)
        output = self.softmax(output)
        return output.view(output.size()[0],output.size()[1])*m,hidden

        

class Encoder(nn.Module):
    def __init__(self,input_size,hidden_size,layer=1,batch_size=16):
        super().__init__()
        self.batch_size = batch_size
        self.layer = layer
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size,num_layers= layer, bidirectional=True)
    def forward(self,input,hidden):
        output = self.linear(input)
        output = F.relu(output)
        
        output , hidden =self.gru(output,hidden)
        
        return output , hidden
    def initHidden(self,batch):
        return torch.zeros(self.layer*2,batch,self.hidden_size)
