import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

MAX_LENGTH = 80
class SequenceTaggle1(nn.Module):
    def __init__(self,num_embeddings, embedding_dim,hidden_size,output_size,device,layer=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.sen_emb = SentenceEncoder(embedding_dim,hidden_size,device,layer=layer)
        self.encoder = Encoder(hidden_size,hidden_size,device,layer=layer)
        self.linear = nn.Linear(hidden_size*2,output_size)
        #self.linear1 = nn.Linear(hidden_size,output_size)
        self.layer = layer
        self.device = device
    def forward(self,input):
        store = torch.tensor([]).to(self.device)
        #store = torch.ones((1,input.size(-1),self.hidden_size))
        for sentence in input: 
            output = self.embedding(sentence)
            hidden = self.sen_emb.initHidden(input.size(-1),self.layer*2)

            cell, hidden = self.sen_emb(output,hidden)
            #print(hidden.size())
            #cell = self.linear1(cell)
            cell =cell[-1]
            cell = cell.view(1,cell.size(0),cell.size(1))
            store = torch.cat((store,cell),dim=0)
        
        #print(store.size())
        output,hidden = self.encoder(store)
        
        #print(output.size())
        output = self.linear(output)
        #print(output.size())
        return output,hidden

class SequenceTaggle(nn.Module):
    def __init__(self,num_embeddings, embedding_dim,hidden_size,output_size,device,layer=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.encoder = Encoder(embedding_dim,hidden_size,device,layer=layer)
        self.linear = nn.Linear(hidden_size*2,output_size)
        #self.linear1 = nn.Linear(hidden_size,output_size)
        self.layer = layer
        self.device = device
    def forward(self,input):

        output = self.embedding(input)
        

        output, hidden = self.encoder(output)

        output = self.linear(output)

        return output,hidden
class SentenceEncoder(nn.Module):
    def __init__(self,input_size,hidden_size,device,layer=1,batch_size=16):
        super().__init__()
        self.input_size = input_size
        self.device = device
        self.encoder = nn.GRU(input_size,hidden_size,num_layers= 1, bidirectional=True)
        self.linear = nn.Linear(hidden_size*2,hidden_size)
        self.hidden_size = hidden_size
        self.LN = nn.LayerNorm(hidden_size*2)
    def forward(self,input,hidden):
        # one sentence of each data in a batch 
        output , hidden = self.encoder(input)
        output = self.LN(output)
        output = self.linear(output)
        
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
        self.linear_new = nn.Linear(2*hidden_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size,num_layers= layer, bidirectional=True)
        init.orthogonal_(self.gru.weight_ih_l0.data)
        init.orthogonal_(self.gru.weight_hh_l0.data)
        self.LN = nn.LayerNorm(hidden_size*2)
        self.gru_F = nn.GRU(hidden_size,hidden_size,num_layers= layer)
        init.orthogonal_(self.gru_F.weight_ih_l0.data)
        init.orthogonal_(self.gru_F.weight_hh_l0.data)
        self.LN_F = nn.LayerNorm(hidden_size)
        self.dropout= nn.Dropout(0.2)
    def forward(self,input):
        output = self.linear(input)
        output = torch.tanh(output)
        hidden = self.initHidden(input.size(1),self.layer)

        gru_output , hidden =self.gru_F(output,hidden)
        gru_output = self.LN_F(gru_output)
        gru_output = self.dropout(gru_output)
        gru_output = torch.cat((gru_output,output),-1)
        gru_output = self.linear_new(gru_output)
        hidden = self.initHidden(input.size(1),self.layer*2)
        output , hidden =self.gru(gru_output,hidden)
        
        output = self.LN(output)
        output = self.dropout(output)
        return output , hidden
    def initHidden(self,batch,layer):
        return torch.zeros(layer,batch,self.hidden_size).to(self.device)

class S2S(nn.Module):
    def __init__(self,num_embeddings, embedding_dim,hidden_size,output_size,device,layer=1,attention=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.encoder = S2SEncoder(embedding_dim,hidden_size,device,layer=layer,attention=attention)
        self.decoder = S2SDecoder(embedding_dim,hidden_size,device,layer=layer,attention=attention)
        self.linear = nn.Linear(hidden_size,output_size)
        self.attention = attention
        #self.linear1 = nn.Linear(hidden_size,output_size)
        self.layer = layer
        self.device = device
        
    def forward(self,input,label):
        output = self.embedding(input)
        
        #print(output.size())
        self.decoder.enc_output,hidden = self.encoder(output)
        label_emb = self.embedding(label)
        output,hidden, att_w = self.decoder(label_emb,hidden)
        
        output = self.linear(output)
        #print(output.size())
        return output,hidden,att_w
class S2SDecoder(nn.Module):
    def __init__(self,input_size,hidden_size,device,layer=1,batch_size=16,attention=False, max_length=MAX_LENGTH):
        super().__init__()
        self.batch_size = batch_size
        self.layer = layer
        self.enc_output = None
        self.hidden_size = hidden_size
        self.device =device
        self.attention = attention
        self.max_length = max_length
        self.linear = nn.Linear(input_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size,num_layers= layer)
        if attention:
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        init.orthogonal_(self.gru.weight_ih_l0.data)
        init.orthogonal_(self.gru.weight_hh_l0.data)
    def forward(self,input,hidden):
        output = self.linear(input)
        output = torch.tanh(output)
        output = torch.relu(output)
        if self.attention:
            #print(hidden.size())
            #attn_weights = F.softmax(self.attn(torch.cat((output[0], hidden[0]), 1)), dim=1)
            #print(self.enc_output.size())
            Q = self.attn(self.enc_output)
            Q = Q.permute(1,0,2)
            K_T = hidden.permute(1,2,0)
            att_weight = F.softmax(torch.bmm(Q,K_T),dim=-1)
            #print(att_weight.size())
            att_ap = torch.sum(torch.bmm(att_weight,hidden.permute(1,0,2)),dim=1)
            att_ap = att_ap.unsqueeze(0)
            #print(att_ap.size())
            
        output,hidden = self.gru(output,hidden)
        if self.attention:
            #combine output and attap
            output = torch.cat((att_ap,output),dim=-1)
            output = self.attn_combine(output)
            output = torch.relu(output)
            return output, hidden , att_weight
        return output, hidden, None
class S2SEncoder(nn.Module):
    def __init__(self,input_size,hidden_size,device,layer=1,batch_size=16,attention=False):
        super().__init__()
        self.batch_size = batch_size
        self.layer = layer
        self.hidden_size = hidden_size
        self.device =device
        self.linear = nn.Linear(input_size,hidden_size)
        self.linear_new = nn.Linear(2*hidden_size,hidden_size)

        self.gru = nn.GRU(hidden_size,hidden_size,num_layers= layer, bidirectional=True)
        init.orthogonal_(self.gru.weight_ih_l0.data)
        init.orthogonal_(self.gru.weight_hh_l0.data)
        self.LN = nn.LayerNorm(hidden_size*2)
        self.gru_F = nn.GRU(hidden_size,hidden_size,num_layers= layer)
        init.orthogonal_(self.gru_F.weight_ih_l0.data)
        init.orthogonal_(self.gru_F.weight_hh_l0.data)
        self.LN_F = nn.LayerNorm(hidden_size)
        self.dropout= nn.Dropout(0.2)
    def forward(self,input):
        
        output = self.linear(input)
        output = torch.tanh(output)
        hidden = self.initHidden(input.size(1),self.layer)

        gru_output , hidden =self.gru_F(output,hidden)
        gru_output = self.LN_F(gru_output)
        gru_output = self.dropout(gru_output)
        gru_output = torch.cat((gru_output,output),-1)
        gru_output = self.linear_new(gru_output)
        hidden = self.initHidden(input.size(1),self.layer*2)
        output , hidden =self.gru(gru_output,hidden)
        #print(hidden[1::2].size())
        hidden_out , hidden_out2 = hidden[::2],hidden[1::2]
        hidden = torch.cat((hidden_out,hidden_out2),dim=2)

        hidden = self.LN(hidden)
        hidden = self.linear_new(hidden)
        hidden = torch.tanh(hidden)
        
        return output,hidden
    def initHidden(self,batch,layer):
        return torch.zeros(layer,batch,self.hidden_size).to(self.device)
