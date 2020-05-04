import torch.nn as nn
import torch.nn.functional as F
import torch


class Answer(nn.Module):
    def __init__(self,input_size=768,answerable_size=1,answer_size=400):
        super().__init__()
        self.ctx_len = answer_size
        self.linear_answerable = nn.Linear(input_size,1)
        
        self.linear_start = nn.Linear(input_size,1)
        self.linear_end = nn.Linear(input_size,1)
        
        
    def forward(self,input):
        
        
        seq, pool = input[0],input[1]
        answerable = self.linear_answerable(pool)
        #answerable = self.linear_answerable(seq[:,0])
        start_input = self.linear_start(seq[:,:self.ctx_len])
        end_input = self.linear_end(seq[:,:self.ctx_len])
        
        return answerable,start_input[:,:,0], end_input[:,:,0]