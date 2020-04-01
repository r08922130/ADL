import numpy as np
import json
class Postprocessing:
    def __init__(self):
        pass
    def select_sentence(self,sentences,intervals,result_dict,result_hist,start,mode='valid',model='m1'):
        
        n = start
        if mode == 'test':
            id_start = 3000000
        else:
            id_start = 2000000
        for k in range(len(sentences)):
            result = []
            max_index = -1
            max_Sum = -1
            num_sen = 0
            if model == 'm1':
                num_sen = len(intervals[k])-1
                for i in range(num_sen):
                    isSummary = sum(sentences[k][intervals[k][i]:intervals[k][i+1]])
                    if max_Sum < isSummary/(intervals[k][i+1]-intervals[k][i]):
                        max_Sum = isSummary/(intervals[k][i+1]-intervals[k][i])
                        max_index = i
                    if isSummary > 0.95*(intervals[k][i+1]-intervals[k][i]):
                        result += [i]
                        result_hist += [i/num_sen]
                if len(result) == 0:
                    result+=[int(max_index)]
            else:
                num_sen = len(sentences[k])
                index = np.argmax(sentences[k])
                result+=[int(index)]
                result_hist += [result/num_sen]
            result_dict+=[{'id' : str(n+id_start),'predict_sentence_index':result }] 
            
            n+=1
        return result_dict,result_hist, n

    def toJson(self,file_name,dic):
        with open(file_name,'w') as f :
            for output in dic:
                json.dump(output,f)
                f.write('\n')
