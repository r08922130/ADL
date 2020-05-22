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
        print("convert to Json.....")
        with open(file_name,'w',encoding='utf8') as f :
            for output in dic:
                json.dump(output,f,ensure_ascii=False)
                f.write('\n')
    def indiesToSentences(self,sentences,result_dict,ids,vocab,tokenizer,mode='valid'):
        #sentences shape (# batches , batch size, seq len)
        n = 0
        
        for batch_ids ,batch in zip(ids,sentences):
            for data_id ,sentence in zip(batch_ids,batch):
                s = self.removeAfterEOS(sentence)
                result = tokenizer.decode(s)#.replace(' £ 1 m','').replace(' £','')
                if len(result) > 0 and result[-1] != '.':
                    result+= ' .'
                result+= ' \n'
                result_dict+=[{'id' : data_id,'predict':result }] 
                n+=1
        return result_dict
    def removeAfterEOS(self,sentence):
        stop = -1
        for i in range(len(sentence)):
            if sentence[i] == 2:
                stop = i
                break
        if sentence[stop-1] == 3:
            stop -=1
        if stop != -1:
            sentence = sentence[:stop]
        if len(sentence)>1:
            new_sen = [sentence[0]]
            for s in sentence[1:]:
                if s != new_sen[-1]:
                    new_sen += [s]
            sentence = new_sen
        """result = []
        [result.append(x) for x in sentence if x not in result]    
        if result[-1]!= 6:
            result.append(6)"""
        return sentence
    
    """def removeSpecificDuplicateSubstring(self,sentence):
        i = 0
        size = 0
        result = []
        for k in range(2,len(sentence))
            if sentence[k] ==  """