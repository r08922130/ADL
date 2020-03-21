import numpy as np
import json
class Postprocessing:
    def __init__(self):
        pass
    def select_sentence(self,sentences,intervals,result_dict,start):
        
        n = start
        for k in range(len(sentences)):
            result = []
            max_l = -1
            max_index = -1
            for i in range(len(intervals[k])-1):
                isSummary = np.sum(sentences[k][intervals[k][i]:intervals[k][i+1]])
                if isSummary > intervals[k][i+1]/2:
                    result += [i]
                if max_l < isSummary/intervals[k][i+1]:
                    max_l =isSummary/intervals[k][i+1]
                    max_index = i
            if len(result) == 0:
                result+=[max_index]
            result_dict+=[{'id' : str(n+3000000),'predict_sentence_index':result }] 
            
            n+=1
        return result_dict, n

    def toJson(self,file_name,dic):
        with open(file_name,'w') as f :
            for output in dic:
                json.dump(output,f)
                f.write('\n')
