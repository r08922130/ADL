import json
from torch.utils.data import Dataset
from transformers import BertForQuestionAnswering, BertTokenizer
#model = BertForQuestionAnswering.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

with open("data/dev.json") as f:
    dev = json.load(f)
    dev = [data for data in dev['data']]
print(dev[0]['paragraphs'][0]['context'])
print(tokenizer.encode(dev[0]['paragraphs'][0]['context']))
class QADataset(Dataset):
    def __init__(self, data, padding=0,
                 max_text_len=300, max_summary_len=80,train=True):
        self.data = data
        self.padding = padding
        self.max_text_len = max_text_len
        self.max_summary_len = max_summary_len
        self.train = train
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.train:
            return {
                'id': self.data[index]['id'],
                'text': self.data[index]['text'][:self.max_text_len],
                'summary': self.data[index]['summary'][:self.max_summary_len],
                'len_text': len(self.data[index]['text']),
                'len_summary': len(self.data[index]['summary']),
                'attention_mask': [True] * min(len(self.data[index]['text']),
                                            self.max_text_len)
            }
        else:
            return {
                'id': self.data[index]['id'],
                'text': self.data[index]['text'][:self.max_text_len],
                'len_text': len(self.data[index]['text']),
                'attention_mask': [True] * min(len(self.data[index]['text']),
                                            self.max_text_len)
            }

    def collate_fn(self, samples):
        batch = {}
        if self.train:
            for key in ['id', 'len_text', 'len_summary']:
                batch[key] = [sample[key] for sample in samples]
            
            for key in ['text', 'summary', 'attention_mask']:
                to_len = max([len(sample[key]) for sample in samples])
                padded = pad_to_len(
                    [sample[key] for sample in samples], to_len, self.padding
                )
                batch[key] = torch.tensor(padded)
        else:
            for key in ['id', 'len_text']:
                batch[key] = [sample[key] for sample in samples]
            
            for key in ['text', 'attention_mask']:
                to_len = max([len(sample[key]) for sample in samples])
                padded = pad_to_len(
                    [sample[key] for sample in samples], to_len, self.padding
                )
                batch[key] = torch.tensor(padded)

        return batch