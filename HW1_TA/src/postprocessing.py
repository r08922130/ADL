import numpy as np
import json
class Postprocessing:
    def __init__(self):
        pass
    def select_sentence(self,sentences,intervals,result_dict,start,mode='valid',model='m1'):
        
        n = start
        if mode == 'test':
            id_start = 3000000
        else:
            id_start = 2000000
        for k in range(len(sentences)):
            result = []
            max_index = -1
            max_Sum = -1
            if model == 'm1':
                for i in range(len(intervals[k])-1):
                    isSummary = sum(sentences[k][intervals[k][i]:intervals[k][i+1]])
                    if max_Sum < isSummary:
                        max_Sum = isSummary
                        max_index = i
                    if isSummary > 0.5*(intervals[k][i+1]-intervals[k][i]):
                        result += [i]
                if len(result) == 0:
                    result+=[int(max_index)]
            else:
            
                index = np.argmax(sentences[k])
                result+=[int(index)]
                if len(result) == 0: 
                    index = np.argmax(sentences[k])
                    result+=[int(index)] 
            result_dict+=[{'id' : str(n+id_start),'predict_sentence_index':result }] 
            
            n+=1
        return result_dict, n

    def toJson(self,file_name,dic):
        with open(file_name,'w') as f :
            for output in dic:
                json.dump(output,f)
                f.write('\n')
    def indiesToSentences(self,sentences,result_dict,vocab,tokenizer,mode='valid'):
        #sentences shape (# batches , batch size, seq len)
        tokenizer = Tokenizer(vocab,True)
        if mode == 'test':
            id_start = 3000000
        else:
            id_start = 2000000
        for batch in sentences:
            for sentence in batch:
                s = self.removeAfterEOS(sentence)
                result = tokenizer.decode(s)
                result_dict+=[{'id' : str(n+id_start),'predict':result }] 
        return result_dict
    def removeAfterEOS(self,sentence):
        stop = -1
        for i in range(len(sentence)):
            if sentence[i] == 2:
                stop = i
                break
        return sentence[:stop]