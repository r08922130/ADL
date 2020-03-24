import numpy as np

class Embedding:
    def __init__(self,file_name,dim=50):
        self.emb_dict = []
        self.index_dict = {}
        self.word_dim = 0
        self.word_num = 0
        
        self.load_model(file_name,dim)

    def load_model(self,file_name,dim=50):
        self.word_dim = dim+3
        with open(file_name) as f:
            l = 0
            for line in f:
                vec = line.split(" ")
                self.emb_dict += [np.append(np.asarray(vec[1:],dtype="float32"),[0,0,0])]
                self.index_dict[vec[0]] = l
                l+=1
            self.word_num = l

        # SOS
        self.emb_dict += [np.asarray([0]*self.word_dim,dtype="float32")]
        self.emb_dict[l][-1] = 1
        self.index_dict['<SOS>'] = l
        l+=1
        # EOS
        self.emb_dict += [np.asarray([0]*self.word_dim,dtype="float32")]
        self.emb_dict[l][-2] = 1
        self.index_dict['<EOS>'] = l
        l+=1
        # PAD
        self.emb_dict += [np.asarray([0]*self.word_dim,dtype="float32")]
        self.emb_dict[l][-3] = 1
        self.index_dict['<PAD>'] = l
        l+=1
        self.word_num+=3
        #print(self.emb_dict['<PAD>'])
         #print(emb_dict["anarchism"])
        

#e = Embedding()
#e.load_model("glove.6B/glove.6B.50d.txt")