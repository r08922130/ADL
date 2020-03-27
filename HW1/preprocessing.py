import json
import nltk
from embedding import Embedding
import numpy as np
#id , summary ,text,sent_bounds,extractive_summary
class Preprocessing:
    def __init__(self):
        super().__init__()
        
    def load_data(self,file_name):
        json_array = []
        with open(file_name) as f:
            for line in f:
                json_array += [json.loads(line)]
        return json_array
    def pad_sentences(self,dic ,batch_size=16,dim=50,modes=['train','valid']):
        
        for mode in modes:
            arr = self.load_data("data/{}.jsonl".format(mode))
            
            batches_x = []
            batch_label = []
            batches_labels = []
            interval = []
            interval_s = []
            n = 0
            max_s =0
            max_d = 0
            docs = []
            for k,data in enumerate(arr):
                internals = data['sent_bounds']
                interval += [internals]
                if mode != 'test':
                    ex_sum =data['extractive_summary']
                    summary = data['summary'] 
                    batch_label += [ex_sum]
                text = data['text']
                sents = []
                labels = []
                now = 0
                for internal in internals:
                    t = text[internal[0]:internal[1]]
                    
                    t = self.tokenize(t)
                    t = ["<SOS>"] + t + ["<EOS>"]
                    max_s = max(max_s,len(t))
                    sents += [t]
                    now +=1
                
                docs += [sents]
                max_d = max(max_d,now)
                

                if k%batch_size ==batch_size-1:
                    if mode != 'test':
                        lab = []
                        for label in batch_label:
                            l = [0]* max_d
                            l[label] = 1
                            lab+= [l]
                        
                        batches_labels += [lab]
                        batch_label = []
                    before_map = self.pad(docs,max_d,max_s)
                    docs =[]
                    max_d=0
                    max_s =0
                    batches_x += [self.map(before_map,dic)]
                    
                    
                    batch_label = []
                    interval_s+= [interval]
                    interval = []
                if k % 800 == 799:
                    print(k+1)
                    print(batches_x[-1].shape)
            before_map = self.pad(docs,max_d,max_s)
            batches_x += [self.map(before_map,dic)]
            interval_s+= [interval]
            if mode != 'test':
                lab=[]
                for label in batch_label:
                    l = [0]* max_d
                    l[label] = 1
                    lab+= [l]
                batches_labels += [lab]
            np.save("data/{}_data_{}.npy".format(mode,dim),np.asarray(batches_x))
            np.save("data/{}_label_{}.npy".format(mode,dim),np.asarray(batches_labels))
            np.save("data/{}_interval_{}.npy".format(mode,dim),np.asarray(interval_s))
            print("Save {}".format(mode))
    def pad(self,docs,max_d,max_s):
        doc = []
        for sens in docs:
            s = 0
            t = []
            for sen in sens:
                s+=1
                t += [sen+(["<PAD>"]*(max_s-len(sen)))]
            t += [["<PAD>"]*max_s]*(max_d-s)
            
            doc +=[t]
        return doc
    def map(self,docs,dic):
        mapped = []
        for sens in docs:
            t = []
            for sen in sens:
                t += [[dic[s] for s in sen]]
            mapped += [t]
        return np.asarray(mapped)
    def batch_data(self,dic = None ,batch_size=16,dim=50,modes=['train','valid']):
        #mode : 'extractive' , 'abstractive'
        e = Embedding("glove.6B.{}d.txt".format(dim),dim=dim)
        myEmbedding = []
        dict_in = {}
        dict_in['<PAD>'] = 0
        dict_in['<SOS>'] = 1
        dict_in['<EOS>'] = 2
        myEmbedding += [e.emb_dict[e.index_dict['<PAD>']]]
        myEmbedding += [e.emb_dict[e.index_dict['<SOS>']]]
        myEmbedding += [e.emb_dict[e.index_dict['<EOS>']]]
        mynum = 3
        for mode in modes:
            arr = self.load_data("data/{}.jsonl".format(mode))
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
                            labels += [0] * (l-1)
                            labels+= [1]
                    inter_start += [l+inter_start[-1]]
                    sents += t
                    now +=1

                embed_sents = []
                for word in sents:
                    if word in e.index_dict.keys():
                        embed_sents += [e.index_dict[word]]
                    else:
                        exist = False
                        for i in range(len(word)):
                            if word[:i] in e.index_dict.keys() and word[i:] in e.index_dict.keys():
                                exist = True
                                embed_sents += [e.word_num]
                                e.emb_dict += [(e.emb_dict[e.index_dict[word[:i]]]+e.emb_dict[e.index_dict[word[i:]]])/2] 
                                e.index_dict[word] = e.word_num
                                e.word_num +=1
                                break
                        if not exist : 
                            embed_sents += [e.word_num]
                            e.emb_dict+= [np.random.randn(e.word_dim)]
                            e.index_dict[word] = e.word_num
                            e.word_num +=1
                    if word not in dict_in.keys():
                        dict_in[word] = mynum
                        myEmbedding += [e.emb_dict[e.index_dict[word]]]
                        mynum+=1
                    embed_sents[-1] = dict_in[word]
                
                if mode != 'test':
                    batch_label += [labels]
                batch_x += [embed_sents] 
                interval += [inter_start]
                n+=1
                if n % batch_size == 0:
                    if mode != 'test':
                        batch_x,batch_label = self.batch_PAD(dict_in,batch_x,batch_label)
                    else:
                        batch_x,batch_label = self.batch_PAD(dict_in,batch_x,None)
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
                batch_x,batch_label = self.batch_PAD(dict_in,batch_x,batch_label)
            else:
                batch_x,batch_label = self.batch_PAD(dict_in,batch_x,None)
            
            print(n+1)
            print(batch_x.shape)
            batches_x +=[batch_x]
            interval_s += [interval]
            if mode != 'test':
                batches_labels += [batch_label]
            np.save("data/{}_data_{}.npy".format(mode,dim),np.asarray(batches_x))
            np.save("data/{}_label_{}.npy".format(mode,dim),np.asarray(batches_labels))
            np.save("data/{}_interval_{}.npy".format(mode,dim),np.asarray(interval_s))

        np.save("embedding.npy",np.asarray(myEmbedding))
        
        json.dump(dict_in,open("dict.json","w"))
    def batch_PAD(self,dict_in, batch,label=None):
        max_l = 0
        #print(np.asarray(np.repeat([e.emb_dict['<PAD>']],3,axis=0),dtype="float32"))
        batch_size = len(batch)
        #print(batch_size)
        for i in range(batch_size):
            max_l = max(max_l,len(batch[i]))
        #print("before"+str(len(label[0])))
        for i in range(batch_size):
            l = len(batch[i])
            batch[i] = np.append(batch[i], np.asarray(np.repeat([dict_in['<PAD>']],max_l-l,axis=0),dtype="float32"),axis=0)
            
            if label != None:   
                label[i] = np.append(label[i],  np.asarray([0]*(max_l-l),dtype="float32"))
        #print("after"+str(len(label[0])))
        return np.asarray(batch,dtype="float32"),np.asarray(label,dtype="float32")

    def tokenize(self,sent):
        t = nltk.word_tokenize(sent)
        
        return [w.lower() for w in t if w not in " ,-."]
    def word_to_index(self,dic,file_name,batch_size=16,dim=300):
        arr = self.load_data(file_name)
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
                inter_start += [l+inter_start[-1]]
                sents += t
                now +=1

            embed_sents = []
            for word in sents:   
                embed_sents += [dic[word]]    
                
            
            batch_x += [embed_sents] 
            interval += [inter_start]
            n+=1
            if n % batch_size == 0:
                
                batch_x,batch_label = self.batch_PAD(dic,batch_x,None)
                if n % 800 == 0:
                    print(n)
                    print(batch_x.shape)
                batches_x +=[batch_x]
                interval_s += [interval]
                batch_x = []
                interval = []
                
        
        
        batch_x,batch_label = self.batch_PAD(dic,batch_x,None)
        
        print(n+1)
        print(batch_x.shape)
        batches_x +=[batch_x]
        interval_s += [interval]
        print(batches_x[0][0])
        return np.asarray(batches_x), np.asarray(interval_s)

#nltk.download('all-corpus')


