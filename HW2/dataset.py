import torch
from torch.utils.data import Dataset


class QADataset(Dataset):
    def __init__(self, data,padding=0,train=True):
        self.data = data
        self.padding = padding
        self.train = train
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.train:
            return {
                'id': self.data[index]['id'],
                'text': self.data[index]['text'],
                'label_answerable': self.data[index]['label_answerable'],
                'label_answer':  self.data[index]['label_answer'],
                'attention_mask' : self.data[index]['attention_mask'],
                'token_type_ids' : self.data[index]['token_type_ids'],
                'SEP' : self.data[index]['SEP']
            }
        else:
            return {
                'id': self.data[index]['id'],
                'text': self.data[index]['text'],
                'attention_mask' : self.data[index]['attention_mask'],
                'token_type_ids' : self.data[index]['token_type_ids'],
                'SEP' : self.data[index]['SEP']
                
            }

    def collate_fn(self, samples):
        batch = {}
        
            
        batch['id'] = [sample['id'] for sample in samples]
        #to_len = max([len(sample['text']) for sample in samples])
        #padded = [sample['text'].tolist()+[self.padding]*(to_len-len(sample['text'])) for sample in samples]
        padded = [sample['text'].tolist() for sample in samples]
        """for sample in samples:
            print(len(sample['text']))"""
        batch['text'] = torch.tensor(padded)
        #batch['token_type_ids'] = [sample['token_type_ids'].tolist() +[1]*(to_len-len(sample['token_type_ids'])) for sample in samples  ]
        batch['token_type_ids'] = [sample['token_type_ids'].tolist()for sample in samples  ]
        #print(batch['token_type_ids'][0])
        batch['token_type_ids'] = torch.tensor(batch['token_type_ids'])
        #batch['token_type_ids'] = torch.ones(batch['token_type_ids'].size()) - batch['token_type_ids']
        #batch['attention_mask'] = [sample['attention_mask'].tolist() +[0]*(to_len-len(sample['attention_mask'])) for sample in samples  ]
        batch['attention_mask'] = [sample['attention_mask'].tolist() for sample in samples  ]
        batch['attention_mask'] = torch.tensor(batch['attention_mask'])
        batch['SEP'] = [sample['SEP'] for sample in samples]
        if self.train:
            
            #print(pads.size())
            batch['label_answer'] = torch.tensor([sample['label_answer'] for sample in samples])
            batch['label_answerable'] = torch.tensor([sample['label_answerable'] for sample in samples])

            

        return batch