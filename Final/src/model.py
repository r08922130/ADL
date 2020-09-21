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
class TagSentenceModel(nn.Module):
    def __init__(self,hidden_dim=768,num_tags=20,attention=False):
        super(TagSentenceModel,self).__init__()
        self.albert = AlbertModel.from_pretrained("ALINEAR/albert-japanese-v2")
        self.dropout = nn.Dropout(0.1)
        self.tags = nn.Linear(768,num_tags)
        self.num_tags = num_tags
    def forward(self,ids):
        ## output : (seq_out , pool_out)
        x = self.albert(input_ids=ids)
        drop_tag = self.dropout(x[1])
        pred_tags = self.tags(drop_tag) # batch * 20
        
        return pred_tags
class SentenceCNNGRUModel(nn.Module):
    def __init__(self,emb_size=768,num_tags=20,attention=False):
        super(SentenceCNNGRUModel,self).__init__()
        self.gru = nn.GRU(emb_size,emb_size, num_layers= 1, bidirectional=True)
        self.conv1 = nn.Conv1d(emb_size,256,kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(256,emb_size,kernel_size=3,padding=1)
        self.linear= nn.Linear(emb_size*2,num_tags)
    def forward(self,doc,init_hidden):
        doc = doc.permute(0,2,1)
        doc = self.conv1(doc)
        doc = F.relu(doc)
        doc = self.conv2(doc)
        doc = doc.permute(2,0,1)
        doc, hidden = self.gru(doc,init_hidden)
        doc = doc.permute(1,0,2)
        doc = self.linear(doc)
        return doc
class Combine_GRU_CNN_Model(nn.Module):
    def __init__(self,emb_size=768,num_tags=20,attention=False):
        super(Combine_GRU_CNN_Model,self).__init__()
        self.g_model = SentenceTaggingModel(emb_size=emb_size,num_tags=num_tags)
        self.c_model = SentenceCNNTaggingModel(emb_size=emb_size, num_tags=num_tags)
        self.linear = nn.Linear(emb_size,num_tags*2)
        self.num_tags = num_tags
    def forward(self,doc,init_hidden):
        g_doc = self.g_model(doc,init_hidden)
        c_doc = self.c_model(doc,init_hidden)
        weight = self.linear(doc)
        doc = torch.cat((g_doc,c_doc),dim=-1)

        doc = doc * weight
        
        doc = doc[:,:,:self.num_tags]+doc[:,:,self.num_tags:]
        return doc
class Combine_GRU_FCN_Model(nn.Module):
    def __init__(self,emb_size=768,num_tags=20,attention=False):
        super(Combine_GRU_FCN_Model,self).__init__()
        self.g_model = SentenceTaggingModel(emb_size=emb_size,num_tags=num_tags,attention=attention)
        self.c_model = SentenceFCNTaggingModel(emb_size=emb_size, num_tags=num_tags,attention=attention)
        self.linear = nn.Linear(emb_size,num_tags*2)
        self.linear_c = nn.Linear(num_tags*2,num_tags)
        self.num_tags = num_tags
    def forward(self,doc,init_hidden):
        g_doc = self.g_model(doc,init_hidden)
        c_doc = self.c_model(doc,init_hidden)
        weight = self.linear(doc)
        doc = torch.cat((g_doc,c_doc),dim=-1)
        
        doc = doc * weight
        
        doc =self.linear_c(doc)
        
        return doc
class Combine_GRU_FCN_Attention_Model(Combine_GRU_FCN_Model):
    def __init__(self,emb_size=768,num_tags=20,attention=False):
        super(Combine_GRU_FCN_Attention_Model,self).__init__(emb_size=emb_size,num_tags=num_tags)
        self.att = SelfAttention(emb_size=emb_size,num_tags=num_tags)
        self.linear = nn.Linear(emb_size,num_tags*3)
        
    def forward(self,doc,init_hidden):
        g_doc = self.g_model(doc,init_hidden)
        c_doc = self.c_model(doc,init_hidden)
        a_doc = self.att(doc,None)
        weight = self.linear(doc)
        doc = torch.cat((g_doc,c_doc,a_doc),dim=-1)
        
        doc = doc * weight
        
        doc = doc[:,:,:self.num_tags]+doc[:,:,self.num_tags:2*self.num_tags]+doc[:,:,2*self.num_tags:]
        
        return doc
class ResidualFCN(nn.Module):
    def __init__(self,emb_size=768,num_tags=20,head=8,attention=False):
        super(ResidualFCN,self).__init__()
        
        self.conv1 = nn.Conv1d(emb_size,256,kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(256,emb_size,kernel_size=3,padding=1)
        self.conv3 = nn.Conv1d(emb_size,num_tags,kernel_size=3,padding=1)
        self.layer_norm = nn.LayerNorm(emb_size)
        #self.linear = nn.Linear(emb_size,num_tags)
        self.attention = attention
        if self.attention:
            self.att = MultiHead(emb_size=emb_size,head=head)
    def forward(self,doc,init_hidden):
        
        if self.attention:
            doc = self.att(doc)
        doc = doc.permute(0,2,1)
        res = doc
        doc = self.conv1(doc)
        doc = F.relu(doc)
        doc = self.conv2(doc)
        doc = F.relu(doc)
        doc = doc.permute(0,2,1)
        doc = self.layer_norm(doc)
        doc = doc.permute(0,2,1)
        doc = res + doc
        doc = self.conv3(doc)
        doc = doc.permute(0,2,1)
        
        return doc
class AttentionModule(nn.Module):
        
    def forward(self,query,key,value):
        
        key = key.permute(0,2,1)
        
        att_weight = F.softmax(torch.bmm(query,key)/(768**(1/2)),dim=-1)
        att = torch.bmm(att_weight,value)
        return att, att_weight
class SentenceGCNModel(nn.Module):
    def __init__(self,emb_size=768,num_tags=20,attention=False):
        super(SentenceGCNModel,self).__init__()
        self.linear = nn.Linear(emb_size,emb_size)
        self.linear_class = nn.Linear(emb_size,num_tags)
    def forward(self,A,I,doc):
        doc = self.linear(doc)
        D = torch.diag((1/(A.sum(1)+1))[0]).unsqueeze(0)
        doc = torch.bmm(D,torch.bmm((A+I),doc))
        doc = self.linear_class(F.relu(doc))
        return doc
class SentenceTaggingModel(nn.Module):
    def __init__(self,emb_size=768,num_tags=20,head=8,attention=False):
        super(SentenceTaggingModel,self).__init__()
        
        self.gru = nn.GRU(emb_size,emb_size, num_layers= 1, bidirectional=True)
        self.linear_1 = nn.Linear(emb_size,emb_size)
        self.linear = nn.Linear(emb_size*2,num_tags)
        self.attention = attention
        if attention:
            self.att = MultiHead(emb_size,head)
    def forward(self,doc,init_hidden):
        
        
        if self.attention:
            
            doc = self.att(doc)
            #doc = torch.cat((att,doc),dim=-1)
        else:
            doc = self.linear_1(doc)
        doc = doc.permute(1,0,2)
        doc,hidden = self.gru(doc)
        #print(doc.size())
        doc = doc.permute(1,0,2)
        doc = torch.tanh(doc)
        doc = self.linear(doc)
        
        return doc
class MultiHead(nn.Module):
    def __init__(self,emb_size,head):
        super(MultiHead,self).__init__()
        
        self.linear_q = nn.Linear(emb_size,emb_size)
        self.linear_k = nn.Linear(emb_size,emb_size)
        self.linear_v = nn.Linear(emb_size,emb_size) 
        self.linear_o = nn.Linear(emb_size,emb_size)
        self.head=head 
        self.layer_norm = nn.LayerNorm(emb_size)
        self.att = AttentionModule()
    def forward(self,doc):
        res = doc
        query = self.linear_q(doc)
        key = self.linear_k(doc)
        value = self.linear_v(doc)
        query = self.reshape_to_batches(query)
        key = self.reshape_to_batches(key)
        value = self.reshape_to_batches(value)
        att, att_w = self.att(query,key,value)
        att = self.reshape_from_batches(att)
        o = self.linear_o(att)
        doc = self.layer_norm(res+o)
        return doc
    def reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head
        return x.reshape(batch_size, seq_len, self.head, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head, seq_len, sub_dim)
    def reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head
        out_dim = in_feature * self.head
        return x.reshape(batch_size, self.head, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)
class SelfAttention(nn.Module):
    def __init__(self,emb_size=768,num_tags=20,head=8,attention=False):
        super(SelfAttention,self).__init__()
        self.multi_head1 = MultiHead(emb_size,head)
        self.multi_head2 = MultiHead(emb_size,head)
        self.linear = nn.Linear(emb_size,num_tags)
    def forward(self,doc,init_hid):
        doc = self.multi_head1(doc)
        doc = self.multi_head2(doc)
        doc = self.linear(doc)
        return doc
    
class SentenceTaggingModel_2(nn.Module):
    def __init__(self,emb_size=768,num_tags=20,head=8,attention=False):
        super(SentenceTaggingModel_2,self).__init__()
        self.gru = nn.GRU(emb_size,emb_size, num_layers= 1, bidirectional=False)
        self.linear_1 = nn.Linear(emb_size,emb_size)
        self.linear = nn.Linear(emb_size,num_tags)
        self.attention = attention
        if attention:
            self.att = MultiHead(emb_size,head)
    def forward(self,doc,init_hidden):
        if self.attention:
            doc = self.att(doc)
        else:
            doc = self.linear_1(doc)
        doc = doc.permute(1,0,2)
        doc,hidden = self.gru(doc)
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
class SentenceFCNTaggingModel(nn.Module):
    def __init__(self,emb_size=768,num_tags=20,head=8,attention=False):
        super(SentenceFCNTaggingModel,self).__init__()
        self.gru = nn.GRU(emb_size,emb_size, num_layers= 1, bidirectional=True)
        self.conv1 = nn.Conv1d(emb_size,256,kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(256,num_tags,kernel_size=3,padding=1)
        #self.linear = nn.Linear(emb_size,num_tags)
        self.attention = attention
        if self.attention:
            self.att = MultiHead(emb_size=emb_size,head=head)
    def forward(self,doc,init_hidden):
        if self.attention:
            doc = self.att(doc)
        doc = doc.permute(0,2,1)
        doc = self.conv1(doc)
        doc = F.relu(doc)
        doc = self.conv2(doc)
        
        doc = doc.permute(0,2,1)
        
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
