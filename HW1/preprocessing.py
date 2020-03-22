import json
import nltk
from embedding import Embedding
import numpy as np
#id , summary ,text,sent_bounds,extractive_summary
class Preprocessing:
    def __init__(self,file_name):
        super().__init__()
        self.file_name = file_name
    def load_data(self):
        json_array = []
        with open(self.file_name) as f:
            for line in f:
                json_array += [json.loads(line)]
        return json_array
    def batch_data(self,batch_size=16,dim=50,mode='train'):
        #mode : 'extractive' , 'abstractive'
        e = Embedding("glove.6B.{}d.txt".format(dim),dim=dim)
        
        arr = self.load_data()
        batch_x = []
        batches_x = []
        batch_label = []
        batches_labels = []
        interval = []
        interval_s = []
        n = 0
        for data in arr:
            #print(n)
            internals = data['sent_bounds']
            if mode != 'test':
                ex_sum =data['extractive_summary']
                summary = data['summary'] 
            text = data['text']
            sents = []
            labels = []
            now = 0
            inter_start = [0]
            for internal in internals:
                t = text[internal[0]:internal[1]]
                
                t = self.tokenize(t)
                t = ["<SOS>"] + t + ["<EOS>"]
                l = len(t)
                if mode != 'test':
                    if now != ex_sum:
                        labels += [0] * l
                    else :
                        labels += [1] * l
                inter_start += [l+inter_start[-1]]
                sents += t
                now +=1

            embed_sents = []
            for word in sents:
                if word in e.emb_dict:
                    embed_sents += [e.emb_dict[word]]
                else:
                    exist = False
                    for i in range(len(word)):
                        if word[:i] in e.emb_dict and word[i:] in e.emb_dict:
                            exist = True
                            embed_sents += [(e.emb_dict[word[:i]]+e.emb_dict[word[i:]])/2]
                            break
                    if not exist : 
                        embed_sents += [np.random.randn(e.word_dim)]
            
            
            if mode != 'test':
                batch_label += [labels]
            batch_x += [embed_sents] 
            interval += [inter_start]
            n+=1
            if n % batch_size == 0:
                if mode != 'test':
                    batch_x,batch_label = self.batch_PAD(e,batch_x,batch_label)
                else:
                    batch_x,batch_label = self.batch_PAD(e,batch_x,None)
                if n % 800 == 0:
                    print(n)
                    print(batch_x.shape)
                batches_x +=[batch_x]
                interval_s += [interval]
                batch_x = []
                interval = []
                if mode != 'test':
                    batches_labels += [batch_label]
                    batch_label = []
        
        if mode != 'test':
            batch_x,batch_label = self.batch_PAD(e,batch_x,batch_label)
        else:
            batch_x,batch_label = self.batch_PAD(e,batch_x,None)
            
        print(n+1)
        print(batch_x.shape)
        batches_x +=[batch_x]
        interval_s += [interval]
        if mode != 'test':
            batches_labels += [batch_label]
        return np.asarray(batches_x), np.asarray(batches_labels),np.asarray(interval_s)
    def batch_PAD(self,e, batch,label=None):
        max_l = 0
        #print(np.asarray(np.repeat([e.emb_dict['<PAD>']],3,axis=0),dtype="float32"))
        batch_size = len(batch)
        #print(batch_size)
        for i in range(batch_size):
            max_l = max(max_l,len(batch[i]))
        #print("before"+str(len(label[0])))
        for i in range(batch_size):
            l = len(batch[i])
            batch[i] = np.append(batch[i], np.asarray(np.repeat([e.emb_dict['<PAD>']],max_l-l,axis=0),dtype="float32"),axis=0)
            
            if label != None:   
                label[i] = np.append(label[i],  np.asarray([0]*(max_l-l),dtype="float32"))
        #print("after"+str(len(label[0])))
        return np.asarray(batch,dtype="float32"),np.asarray(label,dtype="float32")

    def tokenize(self,sent):
        t = nltk.word_tokenize(sent)
        
        return [w.lower() for w in t if w not in " ,-."]
#nltk.download('all-corpus')


