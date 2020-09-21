import torch
import torch.nn as nn
from transformers import AlbertModel

class TagValueModel(nn.Module):
    def __init__(self,hidden_dim=768,num_tags=20):
        super(TagValueModel,self).__init__()
        self.albert = AlbertModel.from_pretrained("ALINEAR/albert-japanese-v2")
        self.dropout = nn.Dropout(0.1)
        self.tags = nn.Linear(768,20)
        #self.tags_insentence = nn.Linear(768,20)
        self.starts = nn.Linear(768,20)
        self.ends = nn.Linear(768,20)
        self.num_tags = num_tags
    def forward(self,ids,type_ids,att_ids,softmax_mask=None):
        ## output : (seq_out , pool_out)
        x = self.albert(input_ids=ids,
                            attention_mask=type_ids,
                            token_type_ids=att_ids.long())
        drop_tag = self.dropout(x[1])
        pred_tags = self.tags(drop_tag) # batch * 20
        drop_tag = self.dropout(x[0])
        #pred_sentence_tags = self.tags_insentence(drop_tag)
        #pred_tags = pred_sentence_tags.mean(1) + pred_tags
        pred_starts = self.starts(x[0]).permute(0,2,1) # batch * 20 * seq_len
        pred_ends = self.ends(x[0]).permute(0,2,1) # batch *20 * seq_len
        if softmax_mask is not None:
            softmax_mask = softmax_mask.repeat(self.num_tags,1,1).permute(1,0,2)
            pred_starts += softmax_mask
            pred_ends += softmax_mask
        return pred_tags, pred_starts, pred_ends 

class SentenceTaggingModel:
    def __init__(self,emb_size=768,num_tags=20):
        self.gru = nn.GRU(emb_size,emb_size, num_layers= 1, bidirectional=True)
        self.linear = nn.Linear(emb_size,num_tags)
    def forward(self,doc):
        doc = self.gru(doc)
        doc = self.linear(doc)
        return doc
