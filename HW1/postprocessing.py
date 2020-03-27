import numpy as np
import json
class Postprocessing:
    def __init__(self):
        pass
    def select_sentence(self,sentences,intervals,result_dict,start,mode='valid'):
        
        n = start
        if mode == 'test':
            id_start = 3000000
        else:
            id_start = 2000000
        for k in range(len(sentences)):
            result = []
            
            """for i in range(len(intervals[k])-1):
                isSummary = sentences[k][intervals[k][i+1]-1]
                if isSummary > 0.5:
                    result += [i]"""
                
            #if len(result) == 0:
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
