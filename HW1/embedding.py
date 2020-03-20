import numpy as np

class Embedding:
    def __init__(self,file_name,dim=50):
        self.emb_dict = {}
        self.word_dim = 0
        self.word_num = 0
        
        self.load_model(file_name,dim)
    def load_model(self,file_name,dim=50):
        self.word_dim = dim+3
        with open(file_name) as f:
            l = 0
            for line in f:
                vec = line.split(" ")
                self.emb_dict[vec[0]] = np.append(np.asarray(vec[1:],dtype="float32"),[0,0,0])
                
                l+=1
            self.word_num = l
        self.emb_dict['<SOS>']= np.asarray([0]*self.word_dim,dtype="float32")
        self.emb_dict['<SOS>'][-1] = 1
        self.emb_dict['<EOS>']= np.asarray([0]*self.word_dim,dtype="float32")
        self.emb_dict['<EOS>'][-2] = 1
        self.emb_dict['<PAD>']= np.asarray([0]*self.word_dim,dtype="float32")
        self.emb_dict['<PAD>'][-3] = 1
        #print(self.emb_dict['<PAD>'])
         #print(emb_dict["anarchism"])
        

#e = Embedding()
#e.load_model("glove.6B/glove.6B.50d.txt")