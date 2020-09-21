import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AlbertModel
class Evaluation_Model(nn.Module):
    def __init__(self,hidden_dim=768,num_tags=20,ckpt_cls="",ckpt_start="",ckpt_end=""):
        super(Evaluation_Model,self).__init__()
        self.num_tags = num_tags
        self.start = TagValueModel(num_tags=num_tags)
        self.start.load_state_dict(torch.load(ckpt_start))
        
        self.end = TagValueModel(num_tags=num_tags)
        self.end.load_state_dict(torch.load(ckpt_end))
        
        self.cls = TagValueModel(num_tags=num_tags)
        self.cls.load_state_dict(torch.load(ckpt_cls))
    def forward(self,ids,type_ids,att_ids,softmax_mask):
        p_tags, _, _, sentence_emb = self.cls(ids,type_ids,att_ids,softmax_mask)
        _, p_starts, _,_ = self.start(ids,type_ids,att_ids,softmax_mask)
        _, _, p_ends,_ = self.end(ids,type_ids,att_ids,softmax_mask)
        return p_tags, p_starts,p_ends
class TagValueModel(nn.Module):
    def __init__(self,hidden_dim=768,num_tags=20):
        super(TagValueModel,self).__init__()
        self.albert = AlbertModel.from_pretrained("ALINEAR/albert-japanese-v2")
        self.dropout = nn.Dropout(0.1)
        self.tags = nn.Linear(768,num_tags)
        #self.tags_insentence = nn.Linear(768,20)
        self.starts = nn.Linear(768,num_tags)
        self.ends = nn.Linear(768,num_tags)
        self.num_tags = num_tags
    def forward(self,ids,type_ids,att_ids,softmax_mask=None):
        ## output : (seq_out , pool_out)
        x = self.albert(input_ids=ids,
                            token_type_ids=type_ids.long(),
                            attention_mask=att_ids,)
        drop_tag = self.dropout(x[1])
        pred_tags = self.tags(drop_tag) # batch * 20
        drop_tag = self.dropout(x[0])
        pred_starts = self.starts(x[0]).permute(0,2,1) # batch * 20 * seq_len
        pred_ends = self.ends(x[0]).permute(0,2,1) # batch *20 * seq_len
        if softmax_mask is not None:
            softmax_mask = softmax_mask.repeat(self.num_tags,1,1).permute(1,0,2)
            pred_starts += softmax_mask
            pred_ends += softmax_mask
        first_sentence = (1-type_ids)* att_ids
        return pred_tags, pred_starts, pred_ends ,self.get_emb(x[0],first_sentence)
    def get_emb(self,input,type_ids):
        
        mask = type_ids.unsqueeze(-1).repeat(1,1,input.size(-1))
        input = input * mask        
        return input.sum(dim=1)/mask.sum(dim=1)
class SentenceTaggingModel(nn.Module):
    def __init__(self,emb_size=768,num_tags=20,attention=False):
        super(SentenceTaggingModel,self).__init__()
        self.gru = nn.GRU(emb_size,emb_size, num_layers= 1, bidirectional=True)
        self.linear_1 = nn.Linear(emb_size,emb_size)
        self.linear = nn.Linear(emb_size*2,num_tags)
        
    def forward(self,doc,init_hidden):
        
        doc = self.linear_1(doc)
        doc = doc.permute(1,0,2)
        doc,hidden = self.gru(doc,init_hidden)
        #print(doc.size())
        doc = doc.permute(1,0,2)
        doc = torch.tanh(doc)
        doc = self.linear(doc)
        
        return doc

class SentenceCNNTaggingModel(nn.Module):
    def __init__(self,emb_size=768,num_tags=20,attention=False):
        super(SentenceCNNTaggingModel,self).__init__()
        self.gru = nn.GRU(emb_size,emb_size, num_layers= 1, bidirectional=True)
        self.conv1 = nn.Conv1d(emb_size,256,kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(256,emb_size,kernel_size=3,padding=1)
        self.linear = nn.Linear(emb_size,num_tags)
        
    def forward(self,doc,init_hidden):
        doc = doc.permute(0,2,1)
        doc = self.conv1(doc)
        doc = F.relu(doc)
        doc = self.conv2(doc)
        
        doc = doc.permute(0,2,1)
        doc = torch.tanh(doc)
        doc = self.linear(doc)
        
        return doc

class Encoder_Decoder(nn.Module):
    def __init__(self,emb_size=768,num_tags=20,attention=False):
        super(Encoder_Decoder,self).__init__()
        self.encoder = Encoder(emb_size=emb_size,attention=attention)
        self.decoder = Decoder(emb_size=emb_size,num_tags=num_tags,attention=attention)
    def forward(self,doc,init_hidden):
        enc_output, hidden = self.encoder(doc,init_hidden)
        doc = self.decoder(doc,hidden,enc_output)
        return doc

class Encoder(nn.Module):
    def __init__(self,emb_size=768,attention=False):
        super(Encoder,self).__init__()
        self.gru = nn.GRU(emb_size,emb_size, num_layers= 1, bidirectional=True)
        self.linear_1 = nn.Linear(emb_size,emb_size)
        self.attention = attention
        
    def forward(self,doc,init_hidden):
        
        doc = doc.permute(1,0,2) # seq len, batch size, hidden size
        doc = self.linear_1(doc)
        output, hidden = self.gru(doc,init_hidden)
        hidden_out , hidden_out2 = hidden[::2],hidden[1::2]
        hidden = torch.cat((hidden_out,hidden_out2),dim=2)
        return output, hidden
class Decoder(nn.Module):
    def __init__(self,emb_size=768,num_tags=20,attention=False):
        super(Decoder,self).__init__()
        self.gru = nn.GRU(emb_size*2,emb_size, num_layers= 1, bidirectional=False)
        self.linear_1 = nn.Linear(emb_size,emb_size*2)
        if not attention:
            self.linear = nn.Linear(emb_size,num_tags)
        else:
            self.linear = nn.Linear(emb_size*2,num_tags)
        self.attention = attention
        if attention:
            self.linear_q = nn.Linear(emb_size,emb_size)
            self.linear_k = nn.Linear(emb_size*2,emb_size)
            self.linear_v = nn.Linear(emb_size*2,emb_size)
    def forward(self,doc,hidden,enc_output):
        
        doc = self.linear_1(doc)
        doc = doc.permute(1,0,2)
        doc,_ = self.gru(doc)
        doc = doc.permute(1,0,2) #batch size, seq len, hidden size
        if self.attention:
            query = self.linear_q(doc)
            #print(query.size())
            key = self.linear_k(enc_output).permute(1,2,0)
            value = self.linear_v(enc_output).permute(1,0,2)
            att_weight = F.softmax(torch.bmm(query,key),dim=-1)
            att = torch.bmm(att_weight,value)
            doc = torch.cat((att,doc),dim=-1)

        doc = self.linear(doc)
        return doc
