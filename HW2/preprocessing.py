

from transformers import BertTokenizer
import torch
import matplotlib.pyplot as plt

#keys 'title' , 'id', 'paragraphs'

#keys in paragraphs :'context','id','qas'  

#keys in qas: 'id'(label id), 'question', 'answer', 'answerable'


class Preprocess:
    def __init__(self,ctx_max_len=400,question_max=40):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese',do_lower_case=True)
        self.ctx_max_len = ctx_max_len
        self.question_max =question_max
    def preprocess_data(self,data,train=True,name='Train'):
        preprocess = []
        mid_start1 = self.ctx_max_len-50
        mid_start2 = 600
        max_ans_len = 20 
        last_start_index1 = -1
        last_start_index2 = -1
        question_max = 100
        ctx_len_hist = []
        q_len_hist = []
        answer_start_hist = []
        answer_start_after_hist = []
        answers_len = []
        max_ans = 0
        for d in data:
            for paragraph in d['paragraphs']:
                context = paragraph['context']
                  
                tokenize_context = self.tokenizer.tokenize(context)
                
                c_l = 0
                ctx_len = len(tokenize_context)
                """if ctx_len >= 800:    
                    tokenize_context = tokenize_context[:800]"""
                ctx_len_hist += [ctx_len]
                tokenize_context = tokenize_context[:self.ctx_max_len]
                #print(ctx_len)
                sen = 1 if ctx_len % mid_start1 != 0 else 0
                ctx =[ tokenize_context[mid_start1*i:mid_start1*i+self.ctx_max_len] for i in range(ctx_len // mid_start1+sen)]
                c_l = len(ctx)
                #d_len +=1
                for Q in paragraph['qas']:
                    # 'id'(label id), 'question', 'answer', 'answerable'
                    Q['question'] = self.tokenizer.tokenize(Q['question'])
                    q_len = len(Q['question'])
                    """if name == 'train' and q_len >= 70:    
                        continue"""
                    q_len_hist += [q_len]
                    Q['question'] = Q['question'][-self.question_max:]
                    # true answer start = answer start+1 (CLS)
                    if train:
                        #answer = Q['answers'][0]
                        for answer in Q['answers']:
                            
                            before_answer = self.tokenizer.tokenize(context[:answer['answer_start']])
                            answer_start = len(before_answer)
                            
                            ans = answer['text']
                            ans = self.tokenizer.encode(ans, add_special_tokens=False)
                            
                            ans_len = len(ans)
                            
                            if Q['answerable']:
                                answers_len +=[ans_len]
                                answer_start_hist +=[answer_start]
                            if answer_start +ans_len >=self.ctx_max_len-2:
                                continue
                            max_ans = max(max_ans,ans_len)
                            if ans_len > max_ans_len:
                                continue
                            
                            
                            
                            """
                            if answer_start > self.ctx_max_len and 2*self.ctx_max_len>ctx_len and answer_start + ans_len< ctx_len-10:
                                c = tokenize_context[-self.ctx_max_len:]
                                answer_start = answer_start-(ctx_len-self.ctx_max_len)
                            elif answer_start+ans_len >= self.ctx_max_len-2 and self.ctx_max_len<=ctx_len:
                                c = tokenize_context[(answer_start-self.ctx_max_len//2):(answer_start+self.ctx_max_len//2)]
                                answer_start = self.ctx_max_len//2
                            
                            else:"""
                            has_ctx = False
                            if Q['answerable']:
                                for l in range(c_l):
                                    if answer_start >= mid_start1*l and answer_start+ans_len< mid_start1*l+self.ctx_max_len-2:
                                        c = ctx[l]
                                        answer_start = answer_start - mid_start1*l
                                        has_ctx = True
                                        break
                            else:
                                c = tokenize_context[:self.ctx_max_len]
                                has_ctx = True
                            
                            if not has_ctx :
                                continue
                            if Q['answerable'] and answer_start <0:
                                print("QQ")
                            """if answer_start+ans_len >= self.ctx_max_len-50 and self.ctx_max_len<=ctx_len:
                                c = tokenize_context[-self.ctx_max_len:]
                                answer_start = answer_start - (ctx_len -self.ctx_max_len)"""
                            #SEP position
                            len_c= len(c)+1

                            input = self.tokenizer.encode_plus(c,Q['question'], max_length=self.ctx_max_len+self.question_max, pad_to_max_length=True, add_special_tokens=True, return_tensors='pt')
                            #answer_start +=1
                            #print(input['token_type_ids'][0])
                            input['input_ids'] = input['input_ids'][0]
                            input['attention_mask'] = input['attention_mask'][0]
                            input['token_type_ids'] = input['token_type_ids'][0]
                            replace = False
                            #print(input['input_ids'][len_c])
                            #print(input['token_type_ids'][len_c])
                            #print(input['attention_mask'][len_c])
                            if not Q['answerable']:
                                answer_start = -1
                                ans_len=-1
                                #print("QQQ")
                            preprocess += [{'id': Q['id'],
                                    'text':input['input_ids'],
                                    'label_answer' : [answer_start,max(answer_start+ans_len-1,-1)],
                                    'label_answerable' : 1 if Q['answerable'] else 0,
                                    'token_type_ids': input['token_type_ids'],
                                    'attention_mask' : input['attention_mask'],
                                    'SEP': len_c}]
                            if Q['answerable']:
                                answer_start_after_hist +=[answer_start ]
                            #if Q['answerable'] and name=='train': 
                            #    print(Q['question'])   
                            #    print(f'Ans Len: {ans_len}, '+self.tokenizer.decode(input['input_ids'][preprocess[-1]['label_answer'][0]:preprocess[-1]['label_answer'][1]+1]))
                    else:
                        for c in ctx:
                            len_c= len(c)+1
                            input = self.tokenizer.encode_plus(c,Q['question'], max_length=self.ctx_max_len+self.question_max, pad_to_max_length=True, add_special_tokens=True, return_tensors='pt')
                            
                            input['input_ids'] = input['input_ids'][0]
                            input['attention_mask'] = input['attention_mask'][0]
                            input['token_type_ids'] = input['token_type_ids'][0]
                            preprocess += [{'id': Q['id'],
                                    'text':input['input_ids'],
                                    'token_type_ids': input['token_type_ids'],
                                'attention_mask' : input['attention_mask'],
                                'SEP': len_c}]
                            break
                            
        print(f"MAX {name} ANS LEN is {max_ans}") 
        plt.figure()
        plt.hist(ctx_len_hist,bins=20)
        plt.savefig(f'CTX_len_{name}.png')
        plt.figure()
        plt.hist(q_len_hist,bins=20)
        plt.savefig(f'Q_len_{name}.png')
        if name != 'test':
            plt.figure()
            plt.hist(answers_len,bins=20,cumulative=True,density=True, histtype='barstacked',rwidth=0.8)
            plt.xlabel("Length")
            plt.ylabel("Count(%)")
            plt.savefig(f'answers_len_{name}.png')
            plt.figure()
            plt.hist(answer_start_hist,bins=20)
            plt.savefig(f'answer_start_{name}.png')
            plt.figure()
            plt.hist(answer_start_after_hist,bins=20)
            plt.savefig(f'answer_start_after_{name}.png')
        return preprocess
