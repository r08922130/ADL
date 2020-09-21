import torch
import torch.nn as nn 
class Solver:
    def __init__(self,device,tokenizer):
        self.device = device
        self.tokenizer = tokenizer
    def train(self,train_data,val_data,model,optimizer,epochs=100,pos_weight=None,scheduler=None):
        """Data Structure

            'file_id' : file_id,
            'index' : key,
            'input_ids' : input_ids,
            'token_type_ids' : token_type_ids,
            'attention_mask' : attention_mask,
            'tags' : tags, 
            'starts' : starts,
            'ends' : ends,
            'softmax_mask' : softmax_mask
        """
        
        criterion_tags = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(self.device))
        criterion_span = nn.CrossEntropyLoss(ignore_index=-1)
        train_len = len(train_data)
        val_len = len(val_data)
        min_loss = 1000000
        min_class_loss = 1000000
        min_start_loss = 1000000
        min_end_loss = 1000000
        for ep in range(epochs):
            total_loss = 0
            class_total_loss = 0
            start_total_loss = 0
            end_total_loss = 0

            model.train()
            for i, data in enumerate(train_data):
                
                # input
                input_ids = data['input_ids'].to(self.device)
                token_type_ids = data['token_type_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device)
                softmax_mask = data['softmax_mask'].to(self.device)

                # groundtruth
                tags = data['tags'].to(self.device)
                starts = data['starts'].to(self.device)
                ends = data['ends'].to(self.device)
                
                #prediction, compute loss, update
                p_tags, p_starts, p_ends = model(input_ids,token_type_ids,attention_mask,softmax_mask)
                loss = criterion_tags(p_tags,tags.float())*100
                class_total_loss += loss.item()
                for t in range(20):
                    start_loss = criterion_span(p_starts[:,t,:],starts[:,t])
                    loss += start_loss
                    end_loss = criterion_span(p_ends[:,t,:],ends[:,t])
                    
                    #debug part
                    if end_loss > 10000:
                        print(data['file_id'])
                        print(data['index'])
                        print(starts[:,t],ends[:,t])
                        print(t)
                        print(self.tokenizer.tokenize(self.tokenizer.decode(input_ids[8])))
                        
                        #print(token_type_ids)
                        #print(softmax_mask)
                    loss += end_loss
                    start_total_loss += start_loss.item()
                    end_total_loss += end_loss.item()
                
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i == 0 or (i+1) % 10 == 0:
                    print(f'Train : Epoch : {ep}, step : {i+1}, Class Loss: {class_total_loss/(i+1):.2f}, Start Loss: {start_total_loss/(i+1):.2f}, End Loss: {end_total_loss/(i+1):.2f}, Total Loss : {total_loss/(i+1):.2f}',end='\r')
                if (i+1) == train_len:

                    print(f'Train : Epoch : {ep}, step : {i+1}, Class Loss: {class_total_loss/(i+1):.2f}, Start Loss: {start_total_loss/(i+1):.2f}, End Loss: {end_total_loss/(i+1):.2f}, Total Loss : {total_loss/(i+1):.2f}',end='\n')
            total_loss = 0
            class_total_loss = 0
            start_total_loss = 0
            end_total_loss = 0

            model.eval()
            with torch.no_grad():
                for i, data in enumerate(val_data):
                    # input
                    input_ids = data['input_ids'].to(self.device)
                    token_type_ids = data['token_type_ids'].to(self.device)
                    attention_mask = data['attention_mask'].to(self.device)
                    softmax_mask = data['softmax_mask'].to(self.device)

                    # groundtruth
                    tags = data['tags'].to(self.device)
                    starts = data['starts'].to(self.device)
                    ends = data['ends'].to(self.device)

                    #prediction, compute loss, update
                    p_tags, p_starts, p_ends = model(input_ids,token_type_ids,attention_mask,softmax_mask)
                    loss = criterion_tags(p_tags,tags.float())
                    class_total_loss += loss.item()
                    for t in range(20):
                        start_loss = criterion_span(p_starts[:,t,:],starts[:,t])
                        loss += start_loss
                        end_loss = criterion_span(p_ends[:,t,:],ends[:,t])
                        
                        loss += end_loss
                        start_total_loss += start_loss.item()
                        end_total_loss += end_loss.item()
                    total_loss += loss.item()
                    if i == 0 or (i+1) % 10 == 0:

                        print(f'Valid : Epoch : {ep}, step : {i+1}, Class Loss: {class_total_loss/(i+1):.2f}, Start Loss: {start_total_loss/(i+1):.2f}, End Loss: {end_total_loss/(i+1):.2f}, Total Loss : {total_loss/(i+1):.2f}',end='\r')
                    if (i+1) == val_len:

                        print(f'Valid : Epoch : {ep}, step : {i+1}, Class Loss: {class_total_loss/(i+1):.2f}, Start Loss: {start_total_loss/(i+1):.2f}, End Loss: {end_total_loss/(i+1):.2f}, Total Loss : {total_loss/(i+1):.2f}',end='\n')
            if scheduler!= None:
                scheduler.step()
            if min_class_loss > class_total_loss/val_len:
                print("Save Class model............")
                min_class_loss = class_total_loss/val_len
                torch.save(model.state_dict(), "ckpt/tags.ckpt")
            if min_start_loss > (start_total_loss)/val_len:
                print("Save Start model............")
                min_start_loss = (start_total_loss)/val_len
                torch.save(model.state_dict(), "ckpt/starts.ckpt")
            if min_end_loss > (end_total_loss)/val_len:
                print("Save End model............")
                min_end_loss = (end_total_loss)/val_len
                torch.save(model.state_dict(), "ckpt/ends.ckpt")
            
                
            


    def test(self,test_data,test_data2,tags_model,starts_model,ends_model,tags_model2,starts_model2,ends_model2,tag,threshold=None,top=3,mode='dev'):
        softmax = nn.Softmax(dim=-1)
        prediction = {}
        tags_model.eval()
        starts_model.eval()
        ends_model.eval()
        tags_model2.eval()
        starts_model2.eval()
        ends_model2.eval()
        test_len = len(test_data)
        min_len_dict = {
            0 : 2,
            1 : 2,
            2 : 7,
            3 : 4,
            4 :4,
            5 : 4,
            6 :  4,
            7 : 4,
            8 : 4,
            9 : 4,
            10: 4,
            11: 4,
            12: 4,
            13: 1,
            14:7,
            15:4,
            16:1,
            17:4,
            18:1,
            19:4,  
            }
        max_len_dict = {
            0 : 40,
            1 : 7,
            2 : 40,
            3 : 40,
            4 :40,
            5 : 40,
            6 :  40,
            7 : 40,
            8 : 40,
            9 : 40,
            10: 40,
            11: 40,
            12: 40,
            13: 40,
            14:40,
            15:40,
            16:40,
            17:40,
            18:40,
            19:40,  
            }
        
        dis_dict = {
            0 : [],
            1 : [],
            2 : [],
            3 : [],
            4 : [],
            5 : [],
            6 :  [],
            7 : [],
            8 : [],
            9 : [],
            10: [],
            11: [],
            12: [],
            13: [],
            14:[],
            15:[],
            16:[],
            17:[],
            18:[],
            19:[],  
        }
        with torch.no_grad():
            for i, (data,data2) in enumerate(zip(test_data,test_data2)):
                # input
                input_ids = data['input_ids'].to(self.device)
                token_type_ids = data['token_type_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device)
                softmax_mask = data['softmax_mask'].to(self.device)
                input_ids2 = data2['input_ids'].to(self.device)
                token_type_ids2 = data2['token_type_ids'].to(self.device)
                attention_mask2 = data2['attention_mask'].to(self.device)
                softmax_mask2 = data2['softmax_mask'].to(self.device)
                text = data['text']
                if mode == 'dev':
                    tags = data['tags'].to(self.device)
                p_tags, _, _ = tags_model(input_ids,token_type_ids,attention_mask,softmax_mask)
                _, p_starts, _ = starts_model(input_ids,token_type_ids,attention_mask,softmax_mask)
                _, _, p_ends = ends_model(input_ids,token_type_ids,attention_mask,softmax_mask)
                p_tags2, _, _ = tags_model2(input_ids,token_type_ids,attention_mask,softmax_mask)
                _, p_starts2, _ = starts_model2(input_ids,token_type_ids,attention_mask,softmax_mask)
                _, _, p_ends2 = ends_model2(input_ids,token_type_ids,attention_mask,softmax_mask)
                p_tags = torch.sigmoid(p_tags)/2+torch.sigmoid(p_tags2)/2
                _,index = p_tags.topk(5)
                #print(index.size())
                tag_mul = torch.zeros_like(p_tags).to(self.device)

                for t, idx in enumerate(index):
                    tag_mul[t,idx] = 1
                #print(tag_mul)
                possible_tag = p_tags# * tag_mul
                #print(p_tags)
                # batch * 20 

                #p_tags = p_tags 
                
                # batch * 20 * seq_len
                p_starts = softmax(p_starts)/2+softmax(p_starts2)/2
                p_ends = softmax(p_ends)/2+softmax(p_ends2)/2
                #print(data['file_id'])
                # 20 tags
                for t in range(20):
                    min_len = min_len_dict[t]
                    max_len = max_len_dict[t]
                    #if t == 5:
                        #print(p_tags[:,t])
                    # Candidate span
                    # Find the most possible one
                    if mode == 'dev':
                        tag_prob = tags[:,t] * p_tags[:,t]
                        dis_dict[t] += tag_prob[tag_prob!=0]
                        """print(f"{t} : " )
                        print(tag_prob[tag_prob!=0])"""
                    for k,isTag in enumerate(p_tags[:,t]):
                        key = data['file_id'][k]+'-'+str(data['index'][k].item())
                        if key not in prediction:
                            prediction[key] = {}
                        if data['index'][k].item() > 30:
                            if t <8:
                                continue
                            isTag = possible_tag[k,t]
                        whole_sentence = self.tokenizer.decode(input_ids[k]).replace(" ","")
                        if t == 1:
                            if '場所' in whole_sentence and p_tags[k,4] > threshold[4]:
                                isTag=1
                        
                        #print(data['index'][k].item())
                        # most 公告日(t == 7) in index 3, can be filtered in postprocessing
                        if isTag > threshold[t] or (data['index'][k].item() == 3 and t ==7):
                            if self.skip_impossible_position(t,data['index'][k].item(),whole_sentence):
                                continue
                            

                            #print(k,t)
                            s = -1
                            e = -1
                            max_p = 0
                            
                            pred_s, start = p_starts[k,t].topk(top)
                            pred_e, end = p_ends[k,t].topk(top)
                            for (ps,s_index) in zip(pred_s,start):
                                for (pe,e_index) in zip(pred_e,end):
                                    if s_index<=e_index :
                                        value = self.tokenizer.decode(input_ids[k,s_index.item():e_index.item()+1])
                                        if max_len<= len(value) or len(value) <= min_len:
                                            continue
                                        if max_p< ps.item()*pe.item() and e_index.item()-s_index.item()<40:
                                            max_p = ps.item()*pe.item()
                                            s = s_index.item()
                                            e = e_index.item()+1
                            if s >= e :
                                continue
                            
                            value = self.tokenizer.decode(input_ids[k,s:e])
                            pre_value = self.tokenizer.decode(input_ids[k,:s])
                            
                            if self.filter(t,value):
                                #print(pre_value)
                                
                                    
                                pre_length  = len(pre_value)-5
                                if '<unk>' in pre_value :
                                    pre_length -=5
                                value = self.match_start_end(value,text[k],max(pre_length-1,0))
                                value = self.remove(t,value)
                                if value == "":
                                    continue
                                #value = self.half2full(value)
                                if tag[t] not in prediction[key]:
                                    prediction[key][tag[t]] = ""
                                
                                prediction[key][tag[t]] += value
                        

                if i % 10 == 0:
                    print(f'{i+1}/{test_len}')
                elif (i+1) == test_len:
                    print(f'{i+1}/{test_len}')
        #print(dis_dict)
        return prediction
    def skip_impossible_position(self,t,index,sentence):
        
        if t == 0 :
            if '件名' in sentence:
                return True
        elif t == 1:
            if index > 25:
                return True
        elif t == 2:
            if '件名' not in sentence or index > 15:
                return True
        elif t ==3:
            if index > 25:
                return True
        elif t == 4:
            if '場所' not in sentence or index > 25:
                return True
        elif t != 14:
            if '電話' in sentence or 'fax' in sentence or '電:話' in sentence or 'tel' in sentence:
                return True
        return False
    def filter(self,tag,value):
        notinanswer = ["@","mail"]
        for trash in notinanswer:
            if trash in value:
                return False
        contain_one_dict = {
            0 : ["平成","令和","年"],
            1 : ['都','道','府','県'],
            2 : [],
            3 : [],
            4 :[],
            5 : ["締結","月"],
            6 :  [],
            7 : [],
            8 : [],
            9 : [],
            10: [],
            11: [],
            12: [],
            13: [],
            14:["-","－"],
            15:[],
            16:[],
            17:[],
            18:[],
            19:[],  
            }
        contain_dict = {
            0 : [],
            1 : [],
            2 : [],
            3 : [],
            4 :[],
            5 : [],
            6 :  ["年","日"],
            7 : ["年","日"],
            8 : ["年","日","時"],
            9 : ["年","日","時"],
            10: ["年","日","時"],
            11: ["年","日","時"],
            12: ["年"],
            13: [],
            14:[],
            15:[],
            16:[],
            17:[],
            18:[],
            19:[],  
            }
        filter_dict = {
            0 : ["@","様式","代表者","前記","kw","fax","電:話","go"],
            1 : ["@","様式","代表者","前記","kw","fax","電:話","go"],
            2 : ["@","様式","代表者","前記","kw","fax","電:話","go"],
            3 : ["年","@","様式","以下","代表者","前記","kw","fax","電:話","go"],
            4 :["年","調達","@","様式","代表者","前記","kw","fax","電:話","go"],
            5 : ["@","様式","代表者","前記","kw","fax","電:話","go"],
            6 :  ["@","様式","代表者","前記","kw","fax","電:話","go"],
            7 : ["@","様式","代表者","前記","kw","fax","電:話","go"],
            8 : ["@","様式","代表者","前記","kw","fax","電:話","go"],
            9 : ["@","様式","代表者","前記","kw","fax","電:話","go"],
            10: ["@","様式","代表者","前記","kw","fax","電:話","go"],
            11: ["@","様式","代表者","前記","kw","fax","電:話","go"],
            12: ["@","様式","代表者","前記","kw","fax","電:話","go"],
            13: ["年","@","様式","代表者","前記","kw","fax","電:話","go"],
            14:["年","@","様式","代表者","前記","kw"],
            15:["年","@","様式","代表者","前記","kw","fax","電:話","go"],
            16:["年","@","様式","代表者","前記","kw","fax","電:話","go"],
            17:["年","@","様式","代表者","前記","kw","fax","電:話","go"],
            18:["年","@","様式","代表者","前記","kw","fax","電:話","go"],
            19:["年","@","様式","代表者","前記","kw","fax","電:話","go"],  
            }
        # at least contain one
        if len(contain_one_dict[tag]) > 0:
            for contain in contain_one_dict[tag]:
                if contain in value:
                    return True
            return False
        # need contain all 
        if len(contain_dict[tag]) > 0:
            for contain in contain_dict[tag]:
                if contain not in value:
                    return False
            return True
        # filter impossible answer
        for f in filter_dict[tag]:
            if f in value:
                return False
        return True
    def match_start_end(self,s,text,pre_length):
        s = self.full2half(s).lower()
        #print(pre_length)
        ch_text = self.full2half(text).lower()
        #print(ch_text)
        start = ch_text.find(s[0],pre_length)
        end = start + len(s)
        if text[start:end] == "":
            print(start,end)
            print(s)
            print(ch_text)
            print(text)
        return text[start:end] if end < len(text) else text[start:]
    def full2half(self,ustring):
        ss = []
        for s in ustring:
            rstring = ""
            for uchar in s:
                inside_code = ord(uchar)
                if inside_code == 12288:  # 全形空格直接轉換
                    inside_code = 32
                elif (inside_code >= 65281 and inside_code <= 65374):  # 全形字元（除空格）根據關係轉化
                    inside_code -= 65248
                rstring += chr(inside_code)
            ss.append(rstring)
        return ''.join(ss)
    def remove(self,tag,value):
        encounters = []
        index = -1
        if tag == 0:
            encounters = ['年']
        elif tag == 1:
            #都道府県
            encounters = ['都','道','府','県']
            
        elif tag == 5 or tag == 6 or tag == 7 :
            encounters = ['日']
        elif tag == 9 or tag == 10 or tag == 11 or tag == 12:
            encounters = ['分']
        for enc in encounters:
            index = value.find(enc)
            if index >= 2 :
                break
        if index > 0:
            value = value[:index+1]
        return value
        