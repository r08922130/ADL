import torch
from torch.utils.data import Dataset

class TagValueDataset(Dataset):
    def __init__(self, data,tokenizer=None,padding=0,train=True,tags_num=0):
        self.data = data
        self.padding = padding
        self.train = train
        self.tokenizer = tokenizer
        self.tags_num = tags_num
    
    def __getitem__(self, index):
        """
            'file_id' : file_id,
            'index' : key,
            'input_ids' : input_ids,
            'token_type_ids' : token_type_ids,
            'attention_mask' : attention_mask,
            'tags' : tags,
            'starts' : starts,
            'ends' : ends,
            'softmax_mask' : softmax_mask
        """
        if self.train:
            return {
                'file_id' : self.data[index]['file_id'],
                'index' : self.data[index]['index'],
                'input_ids' : self.data[index]['input_ids'],
                'token_type_ids' : self.data[index]['token_type_ids'],
                'attention_mask' : self.data[index]['attention_mask'],
                'tags' : self.data[index]['tags'],
                'starts' : self.data[index]['starts'],
                'ends' : self.data[index]['ends'],
                'softmax_mask' : self.data[index]['softmax_mask'],
                'text' : self.data[index]['text']
                
            }
        else:
            return {
                'file_id' : self.data[index]['file_id'],
                'index' : self.data[index]['index'],
                'input_ids' : self.data[index]['input_ids'],
                'token_type_ids' : self.data[index]['token_type_ids'],
                'attention_mask' : self.data[index]['attention_mask'],
                'softmax_mask' : self.data[index]['softmax_mask'],
                'text' : self.data[index]['text']
                
                
            }
    def __len__(self):
        return len(self.data)

class DocumentDataset(Dataset):
    def __init__(self,data,graph=None):
        self.data = data
        self.graph= graph
    def __getitem__(self,index):
        """
        data[0] : every sentence in document
        data[1] : every tag in document
        data[2] : document id

        """
        
        return {'embs' :self.data[0][index],'tags': self.data[1][index],'file_id':self.data[2][index]}
    def __len__(self):
        return len(self.data[0])