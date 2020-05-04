import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
class Solver:
    def __init__(self,device,tokenizer=None):
        self.device = device
        self.tokenizer = tokenizer
    def train(self,train,dev,bert_model, model,scheduler=None,epochs=50,lr=0.000001,ctx_max_len=400):
        
        """
            data keys:
                'id'
                'text'
                'label_answerable'
                'label_answer'
                'attention_mask' 
                'token_type_ids' 
        """
        bert_opt=optim.AdamW(bert_model.parameters(), lr=lr)
        min_loss = 100000000
        model_opt=optim.AdamW(model.parameters(), lr=lr)
        min_class_loss = 1000000
        min_start_loss = 1000000
        min_end_loss = 1000000
        train_len = len(train)
        dev_len = len(dev)
        #weights = torch.tensor([1]*ctx_max_len).float()
        #weights[0] = 0.00001
        #print(weights)
        criterion = nn.CrossEntropyLoss(ignore_index=-1).to(self.device)
        """pos_weight=torch.tensor([0.4])"""
        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.4])).to(self.device)
        # no truncate bce loss
        #bce_criterion = nn.BCEWithLogitsLoss().to(self.device)

        for ep in range(epochs):
            #train
            total_loss = 0
            class_total_loss = 0
            start_total_loss = 0
            end_total_loss = 0
            
            model.train()
            bert_model.train()
            for i, data in enumerate(train):
                
                bert_opt.zero_grad()
                model_opt.zero_grad()
                # (batch size , seq_len)
                input_text = data['text'].to(self.device)
                #input_text = input_text.permute(1,0)
                #print(self.tokenizer.decode(data['text'][0]))
                #(batch size, seq_len)
                input_attmask = data['attention_mask'].to(self.device)
                #print(input_attmask[0])
                #input_attmask = input_attmask.permute(1,0)
                
                #print(pad_index)
                #(batch size, seq_len)
                input_token_ids = data['token_type_ids'].to(self.device)
                linear_mask = 1-data['token_type_ids'].to(self.device)
                #print(data['token_type_ids'][0])
                linear_mask  = linear_mask * data['attention_mask'].to(self.device)
                #print(linear_mask)
                #input_token_ids = input_token_ids.permute(1,0)
                
                
                #print(self.tokenizer.decode(data['text'][0][data['label_answer'][0][0].item():data['label_answer'][0][1].item()+1]))
                #print(data['label_answerable'][0])
                total_answerable = torch.sum( data['label_answerable'])
                
                #(batch size)
                label_answerable = data['label_answerable'].to(self.device)
                
                #(batch size, output size)
                label_answer = data['label_answer'].to(self.device)
                
                #print(label_answer.size())
                #label_answer = label_answer.permute(1,0)                
                bert_output = bert_model(input_ids=input_text,
                            attention_mask=input_attmask,
                            token_type_ids=input_token_ids.long())
                #print(bert_output[0].size())
                #print(bert_output[1].size())
                pad_index = (1-linear_mask[:,:ctx_max_len])*0
                total_answer = len(data['text'])
                for k in range(total_answer):
                    
                    
                    #SEP
                    pad_index[k][data['SEP'][k]:] =1e9
                pad_index = pad_index.to(self.device)
                pred_answerable,pred_start,pred_end = model(bert_output)
                
                #pred_start,pred_end = bert_output
                #pred_start,pred_end = pred_start[:ctx_max_len].permute(1,0),pred_end[:ctx_max_len].permute(1,0)
                loss = 0
                """if total_answerable != 0 and total_answerable != total_answer:
                    bce_criterion = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=torch.tensor([(total_answer-total_answerable)/total_answerable])).to(self.device)"""
                """else:
                    bce_criterion = nn.BCEWithLogitsLoss().to(self.device)"""
                
                class_loss = bce_criterion(pred_answerable[:,0],label_answerable.float())
                #print(class_loss)
                #print(pred_start-pad_index)
                start_end_loss = []
                #pred = [pred_start,pred_end]
                pred_start -= pad_index
                pred_end -= pad_index
                #print(torch.softmax(pred_start,dim=1)[0])
                #print(label_answer[:,0])
                #print(label_answer[:,1])
                start_loss = criterion(pred_start[:,1:],label_answer[:,0])
                end_loss = criterion(pred_end[:,1:],label_answer[:,1])
                """for t in range(len(pred)):
                    bert_loss = 0
                    #print(i)
                    berts = pred[t]
                    for j in range(len(berts)):
                        if label_answerable[j]:
                            bert_loss += criterion(berts[j:j+1],(label_answer[j:j+1,t]-1))
                        
                    start_end_loss +=[bert_loss/total_answerable]"""
                    
                class_total_loss += class_loss.item()
                start_total_loss += start_loss.item()
                end_total_loss += end_loss.item()
                loss = start_loss+end_loss+class_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                torch.nn.utils.clip_grad_norm_(bert_model.parameters(),5)
                model_opt.step()
                bert_opt.step()
                total_loss += loss.item()
                if i == 0 or (i+1) % 10 == 0:
                    #print(pred_answerable)
                    #print(label_answerable)
                    print(f'P,G [0] ={torch.sigmoid(pred_answerable[0]).item()}, {label_answerable[0].item() }, Train : Epoch : {ep}, step : {i+1}, Class Loss: {class_total_loss/(i+1):.2f}, Start Loss: {start_total_loss/(i+1):.2f}, End Loss: {end_total_loss/(i+1):.2f}, Total Loss : {total_loss/(i+1):.2f}',end='\r')
                if (i+1) == train_len:
                    #print(pred_answerable)
                    print(f'Train : Epoch : {ep}, step : {i+1}, Class Loss: {class_total_loss/(i+1)}, Start Loss: {start_total_loss/(i+1)}, End Loss: {end_total_loss/(i+1)}, Total Loss : {total_loss/(i+1)}',end='\n')
    
            #valid
            model.eval()
            bert_model.eval()
            val_loss = 0
            class_total_loss = 0
            start_total_loss = 0
            end_total_loss = 0
            with torch.no_grad():
                for i, data in enumerate(dev):
                    # (batch size , seq_len)
                    #print(data['label_answer'][0][0].item())

                    #print(self.tokenizer.decode(data['text'][0][data['label_answer'][0][0].item():data['label_answer'][0][1].item()+1]))
                    #print(data['id'][0])
                    input_text = data['text'].to(self.device)
                    

                    #(batch size, seq_len)
                    input_attmask = data['attention_mask'].to(self.device)
                    
                    total_answer = len(data['text'])
                    
                    #(batch size, seq_len)
                    #(batch size, seq_len)
                    input_token_ids = data['token_type_ids'].to(self.device)
                    
                    
                    total_answerable = torch.sum( data['label_answerable'])
                    
                    #(batch size)
                    label_answerable = data['label_answerable'].to(self.device)
                    
                    #(batch size, output size)
                    label_answer = data['label_answer'].to(self.device)
                    #print(label_answer.size())
                    #label_answer = label_answer.permute(1,0)                
                    bert_output = bert_model(input_ids=input_text,
                                attention_mask=input_attmask,
                                token_type_ids=input_token_ids.long())
                    #pred_start,pred_end = model(bert_output)
                    linear_mask = 1-data['token_type_ids'].to(self.device)
                    #print(data['token_type_ids'][0])
                    linear_mask  = linear_mask * data['attention_mask'].to(self.device)
                    #print(linear_mask)
                    
                    pad_index = (1-linear_mask[:,:ctx_max_len])*0
                    total_answer = len(data['text'])
                    for k in range(total_answer):
                        
                        
                        #SEP
                        pad_index[k][data['SEP'][k]:] =1e9
                    pad_index = pad_index.to(self.device)
                    pred_answerable,pred_start,pred_end = model(bert_output)
                    #pred_start,pred_end = bert_output
                    #pred_start,pred_end = pred_start[:ctx_max_len].permute(1,0),pred_end[:ctx_max_len].permute(1,0)
                    loss = 0
                    
                    
                    class_loss = bce_criterion(pred_answerable[:,0],label_answerable.float())
                    loss = 0
                    """if total_answerable != total_answer:
                        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(total_answer-total_answerable)/total_answerable])).to(self.device)
                    else:
                        bce_criterion = nn.BCEWithLogitsLoss().to(self.device)
                    
                    class_loss = bce_criterion(pred_answerable.squeeze(),label_answerable.float())"""
                    #print(class_loss)
                    pred_start -= pad_index
                    pred_end -= pad_index
                    start_loss = criterion(pred_start[:,1:],label_answer[:,0])
                    end_loss = criterion(pred_end[:,1:],label_answer[:,1])
                    """for t in range(len(pred)):
                        bert_loss = 0
                        #print(i)
                        berts = pred[t]
                        for j in range(len(berts)):
                            if label_answerable[j]:
                                bert_loss += criterion(berts[j:j+1],(label_answer[j:j+1,t]-1))
                            
                        start_end_loss +=[bert_loss/total_answerable]"""
                        
                    class_total_loss += class_loss.item()
                    start_total_loss += start_loss.item()
                    end_total_loss += end_loss.item()
                    loss = start_loss+end_loss+class_loss
                    val_loss += loss.item()
                    if i == 0 or (i+1) % 10 == 0:
                    
                        print(f'P,G [0] ={torch.sigmoid(pred_answerable[0]).item()}, {label_answerable[0].item() }, Valid : Epoch : {ep}, step : {i+1}, Class Loss: {class_total_loss/(i+1):.2f}, Start Loss: {start_total_loss/(i+1):.2f}, End Loss: {end_total_loss/(i+1):.2f}, Total Loss : {val_loss/(i+1):.2f}',end='\r')
                    if (i+1) == dev_len:
                        #print(pred_answerable)
                        print(f'Valid : Epoch : {ep}, step : {i+1}, Class Loss: {class_total_loss/(i+1)}, Start Loss: {start_total_loss/(i+1)}, End Loss: {end_total_loss/(i+1)}, Total Loss : {val_loss/(i+1)}',end='\n')

            val_loss/= (i+1)
            if min_class_loss > class_total_loss/dev_len:
                print("Save Class model............")
                min_class_loss = class_total_loss/dev_len
                torch.save(model.linear_answerable.state_dict(), "ckpt/best_class_notru.ckpt")
            if min_start_loss > (start_total_loss)/dev_len:
                print("Save Start model............")
                min_start_loss = (start_total_loss)/dev_len
                torch.save(model.linear_start.state_dict(), "ckpt/best_linear_start_notru.ckpt")
            if min_end_loss > (end_total_loss)/dev_len:
                print("Save End model............")
                min_end_loss = (end_total_loss)/dev_len
                torch.save(model.linear_end.state_dict(), "ckpt/best_linear_end_notru.ckpt")
            if min_loss > val_loss:
                print("Save Bert model............")
                min_loss = val_loss
                torch.save(bert_model.state_dict(), "ckpt/best_bert_notru.ckpt")
            
    def test(self,test,bert_model, model,scheduler=None,ans_threshold=0.988,ctx_max_len=400,top=3):
        prediction={}
        prediction_ans={}
        bert_model.eval()
        model.eval()
        softmax = nn.Softmax(dim=-1)
        for i, data in enumerate(test):
            input_text = data['text'].to(self.device)

            #(batch size, seq_len)
            input_attmask = data['attention_mask'].to(self.device)
            linear_mask = 1-data['token_type_ids'].to(self.device)
                #print(data['token_type_ids'][0])
            linear_mask  = linear_mask * data['attention_mask'].to(self.device)
            #print(linear_mask)
            
            pad_index = (1-linear_mask[:,:ctx_max_len])*1e9
            total_answer = len(data['text'])
            pad_index[:,0] = 1e9
            #SEP
            for k in range(total_answer):
                #SEP
                pad_index[k][data['SEP'][k]] =1e9
            
            pad_index = pad_index.to(self.device)
            
            
            
            #(batch size, seq_len)
            input_token_ids = data['token_type_ids'].to(self.device)
            total_seq_len = input_token_ids.sum(dim=0)

            bert_output = bert_model(input_ids=input_text,
                            attention_mask=input_attmask,
                            token_type_ids=input_token_ids.long())
            pred_answerable,pred_start,pred_end = model(bert_output)
            """pred_start,pred_end = bert_output
            pred_start,pred_end = pred_start[:ctx_max_len].permute(1,0),pred_end[:ctx_max_len].permute(1,0)"""

            #print(data['text'])
            pred_answerable = torch.sigmoid(pred_answerable)
            #print(pred_answerable)
            pred_start = softmax((pred_start-pad_index)[:,1:])
            pred_end = softmax((pred_end-pad_index)[:,1:])
            #print(sum(pred_start[0]))
            #print(pred_end)
            unanswerables = pred_answerable.squeeze()
            #print(unanswerables)
            for k in range(len(unanswerables)):
                
                if unanswerables[k] > ans_threshold:
                    #print(pred_start[k].topk(1))
                    s = -1
                    e = -1
                    max_p = 0
                    
                    pred_s, start = pred_start[k].topk(top)
                    pred_e, end = pred_end[k].topk(top)
                    for (ps,s_index) in zip(pred_s,start):
                        for (pe,e_index) in zip(pred_e,end):
                            if s_index<=e_index and max_p< ps.item()*pe.item() and e_index.item()-s_index.item()<30:
                                max_p = ps.item()*pe.item()
                                s = s_index.item()
                                e = e_index.item()+1
                    #s = start.item()
                    #e = end.item()+1
                    #print( pred_end[k][start+1:].size())
                    """for (p,index) in zip(pred_s,start):
                        #print(index)
                        if index > ctx_max_len-3:
                            continue
                        pred, end = pred_end[k][index+1:].topk(1)
                        if max_p< pred.item()*p and end.item()<20:
                            max_p = pred.item()*p
                            s = index.item()
                            e = end.item()+s+1+1"""
                    if s >= e :
                        if data['id'][k] not in prediction:
                            prediction[data['id'][k]] = "1"
                            prediction_ans[data['id'][k]] = 0
                        continue
                    if data['id'][k] in prediction and prediction_ans[data['id'][k]] < unanswerables[k].item()*max_p:
                        prediction[data['id'][k]] = self.tokenizer.decode(data['text'][k,s+1:e+1],skip_special_tokens=True).replace(" ","")
                        prediction_ans[data['id'][k]] = unanswerables[k].item()*max_p
                        #print(prediction[data['id'][k]])
                    elif data['id'][k] not in prediction:
                        prediction[data['id'][k]] = self.tokenizer.decode(data['text'][k,s+1:e+1],skip_special_tokens=True).replace(" ","")
                        prediction_ans[data['id'][k]] = unanswerables[k].item()*max_p
                    
                else:
                    if data['id'][k] not in prediction:
                        prediction[data['id'][k]] = ""
                        prediction_ans[data['id'][k]] = 0
            
            if (i+1) % 10 == 0:
                print(f'{i+1} / {len(test)}')
        return prediction