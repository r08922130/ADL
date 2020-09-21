import torch
import torch.nn as nn 
class Solver:
    def __init__(self,device,tokenizer):
        self.device = device
        self.tokenizer = tokenizer
    def train(self,train_data,val_data,model,optimizer,epochs=50,pos_weight=None,scheduler=None,part="",ver="",only_class=False):
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
            'text' : original sentence
        """
        
        criterion_tags = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(self.device))
        criterion_span = nn.CrossEntropyLoss(ignore_index=-1)
        train_len = len(train_data)
        val_len = len(val_data)
        min_loss = 1000000
        min_class_loss = 1000000
        min_start_loss = 1000000
        min_end_loss = 1000000
        num_tags = model.num_tags
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
                p_tags, p_starts, p_ends, sentence_emb = model(input_ids,token_type_ids,attention_mask,softmax_mask)
                loss = criterion_tags(p_tags,tags.float())*num_tags
                class_total_loss += loss.item()
                if not only_class:
                    for t in range(num_tags):
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
                    p_tags, p_starts, p_ends,sentence_emb = model(input_ids,token_type_ids,attention_mask,softmax_mask)
                    loss = criterion_tags(p_tags,tags.float())
                    class_total_loss += loss.item()
                    if not only_class:
                        for t in range(num_tags):
                            start_loss = criterion_span(p_starts[:,t,:],starts[:,t])
                            loss += start_loss
                            end_loss = criterion_span(p_ends[:,t,:],ends[:,t])
                            
                            loss += end_loss
                            start_total_loss += start_loss.item()
                            end_total_loss += end_loss.item()
                    total_loss += loss.item()
                    if i == 0 or (i+1) % 10 == 0:

                        print(f'Validation : Epoch : {ep}, step : {i+1}, Class Loss: {class_total_loss/(i+1):.2f}, Start Loss: {start_total_loss/(i+1):.2f}, End Loss: {end_total_loss/(i+1):.2f}, Total Loss : {total_loss/(i+1):.2f}',end='\r')
                    if (i+1) == val_len:

                        print(f'Validation : Epoch : {ep}, step : {i+1}, Class Loss: {class_total_loss/(i+1):.2f}, Start Loss: {start_total_loss/(i+1):.2f}, End Loss: {end_total_loss/(i+1):.2f}, Total Loss : {total_loss/(i+1):.2f}',end='\n')
            if scheduler!= None:
                scheduler.step()
            if min_class_loss > class_total_loss/val_len:
                print("Save Class model............")
                min_class_loss = class_total_loss/val_len
                torch.save(model.state_dict(), f"ckpt_{ver}/{part}tags.ckpt")
            if min_start_loss > (start_total_loss)/val_len and not only_class:
                print("Save Start model............")
                min_start_loss = (start_total_loss)/val_len
                torch.save(model.state_dict(), f"ckpt_{ver}/{part}starts.ckpt")
            if min_end_loss > (end_total_loss)/val_len and not only_class:
                print("Save End model............")
                min_end_loss = (end_total_loss)/val_len
                torch.save(model.state_dict(), f"ckpt_{ver}/{part}ends.ckpt")
            
                
            
    def two_stage_train(self,train_data,val_data,model,optimizer,epochs=50,pos_weight=None,name="gru",part="",ver="",gcn=False):
        criterion_tags = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(self.device))
        total_loss = 0 
        min_loss = 10000
        train_len = len(train_data)
        val_len = len(val_data)
        print(train_len)
        for ep in range(epochs):
            model.train()
            total_loss = 0 
            for i, data in enumerate(train_data):
                """
                embs : contain sentence embedding in one document
                tags : contain tags per sentence
                """
                
                
                optimizer.zero_grad()
                embs = data['embs'].to(self.device)
                tags = data['tags'].to(self.device)
                #A = data['graph'].to(self.device)
                #I = torch.eye(A.size(-1)).unsqueeze(0).to(self.device)
                init_hidden = torch.zeros(1,1,768).to(self.device)
                if gcn:
                    p_tags = model(A,I,embs)
                
                else:
                    p_tags = model(embs,init_hidden)
                
                loss = criterion_tags(p_tags, tags.float())
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                if (i+1) == train_len:
                    print(f'Train : Epoch : {ep}, step : {i+1}, Total Loss : {total_loss/(i+1):.2f}',end='\n')
                elif i %10 == 0:
                    print(f'Train : Epoch : {ep}, step : {i+1}, Total Loss : {total_loss/(i+1):.2f}',end='\r')
                    
            
            model.eval()
            total_loss = 0 
            with torch.no_grad():
                for i, data in enumerate(val_data):
                    """
                    embs : contain sentence embedding in one document
                    tags : contain tags per sentence
                    """
                    embs = data['embs'].to(self.device)
                    tags = data['tags'].to(self.device)
                    #A = data['graph'].to(self.device)
                    #
                    # I = torch.eye(A.size(-1)).unsqueeze(0).to(self.device)
                    init_hidden = torch.zeros(1,1,768).to(self.device)
                    if gcn:
                        p_tags = model(A,I,embs)
                    else:
                        p_tags = model(embs,init_hidden)
                    loss = criterion_tags(p_tags, tags.float())
                    total_loss += loss.item()
                    

                    if (i+1) == val_len:
                        print(f'Validation : Epoch : {ep}, step : {i+1}, Total Loss : {total_loss/(i+1):.2f}',end='\n')
                    elif i %10 == 0:
                        print(f'Validation : Epoch : {ep}, step : {i+1}, Total Loss : {total_loss/(i+1):.2f}',end='\r')
            if min_loss > total_loss/val_len:
                print("Save Second Layer Class model............")
                min_loss = total_loss/val_len
                torch.save(model.state_dict(), f"ckpt_{ver}/{name}_{part}tags_new.ckpt")
    def extract_sentence_embedding(self,dataset,emb_model):
        emb_model.eval()
        embedding_dict = {}
        data_len = len(dataset)
        with torch.no_grad() : 
            for i, data in enumerate(dataset):
                # input
                input_ids = data['input_ids'].to(self.device)
                token_type_ids = data['token_type_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device)
                softmax_mask = data['softmax_mask'].to(self.device)

                
                #prediction, compute loss, update
                p_tags, p_starts, p_ends, sentence_emb = emb_model(input_ids,token_type_ids,attention_mask,softmax_mask)
                # groundtruth
                if 'tags' in data.keys():
                    tags = data['tags'].to(self.device)
                else:
                    tags = torch.zeros(sentence_emb.size(0))
                
                for b in range(sentence_emb.size(0)):
                    key = data['file_id'][b]+'-'+str(data['index'][b].item())
                    embedding_dict[key] = [sentence_emb[b].cpu(),tags[b].cpu()]
                if i %10 == 0:
                    print(f'{i}/{data_len}',end='\r')

        return self.gather_embedding_document(embedding_dict)
    def gather_embedding_document(self,emb_dict):
        prev_key = ""
        document = {}
        tag_doc = {}
        doc_id = {}
        doc_num = -1
        for key, item in emb_dict.items():
            sen_emb, tag = item
            if prev_key != key[:9]:
                doc_num +=1
                document[doc_num] = sen_emb.unsqueeze(0)
                tag_doc[doc_num] = tag.unsqueeze(0)
                prev_key = key[:9]
                doc_id[doc_num] = key[:9]
                
                continue
            document[doc_num] = torch.cat((document[doc_num],sen_emb.unsqueeze(0)),dim=0)
            tag_doc[doc_num] = torch.cat((tag_doc[doc_num],tag.unsqueeze(0)),dim=0)
            
            prev_key = key[:9]
        
        return document, tag_doc, doc_id


            
                
    def test(self,test_data,model,tag,test_emb=None,two_stage_model=None,threshold=None,top=3,mode='dev',tag_top=4,use_emb=False,middle=False):
        
        softmax = nn.Softmax(dim=-1)
        prediction = {}
        model.eval()
        test_len = len(test_data)
        num_tags = model.num_tags
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
            14:8,
            15:4,
            16:1,
            17:4,
            18:1,
            19:4,  
            }
        max_len_dict = {
            0 : 7,
            1 : 7,
            2 : 50,
            3 : 40,
            4 :40,
            5 : 40,
            6 :  40,
            7 : 40,
            8 : 40,
            9 : 40,
            10: 40,
            11: 25,
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
        collect_emb_pred = torch.zeros((1,num_tags))
        cur_index = 0
        with torch.no_grad():
            if two_stage_model != None:
                two_stage_model.eval()
                for i, data in enumerate(test_emb):
                    embs = data['embs'].to(self.device)
                    init_hidden = torch.zeros(1,1,768).to(self.device)
                    p_tags = two_stage_model(embs,init_hidden)
                    p_tags = torch.sigmoid(p_tags)
                    collect_emb_pred = torch.cat((collect_emb_pred,p_tags.squeeze(0).cpu()),dim=0)
            collect_emb_pred = collect_emb_pred[1:]
                   
            for i, data in enumerate(test_data):
                # input
                
                
                input_ids = data['input_ids'].to(self.device)
                token_type_ids = data['token_type_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device)
                softmax_mask = data['softmax_mask'].to(self.device)
                text = data['text']

                
                
                
                if mode == 'dev':
                    tags = data['tags'].to(self.device)
                p_tags, p_starts, p_ends = model(input_ids,token_type_ids,attention_mask,softmax_mask)
                

                p_tags = torch.sigmoid(p_tags)
                
                _,index = p_tags.topk(tag_top)
                
                #print(index.size())
                tag_mul = torch.zeros_like(p_tags).to(self.device)

                for t, idx in enumerate(index):
                    tag_mul[t,idx] = 1
                #print(tag_mul)
                possible_tag = p_tags * tag_mul
                if two_stage_model != None:
                    # load pred from embedding prediction prob
                    batch_size = input_ids.size(0)
                    emb_pred = (collect_emb_pred[cur_index:cur_index+batch_size].clone().detach()).to(self.device)
                    cur_index += batch_size
                    _,emb_idx = emb_pred.topk(tag_top)
                    tag_mul = torch.zeros_like(emb_pred).to(self.device)
                    for t, idx in enumerate(emb_idx):
                        tag_mul[t,idx] = 1
                    p_tags = p_tags*emb_pred
                    possible_tag = possible_tag *emb_pred* tag_mul
                    p_tags = p_tags**(1/2)
                    possible_tag = possible_tag **(1/2)
                    """p_tags = (p_tags+emb_pred)
                    possible_tag = (possible_tag + emb_pred* tag_mul)
                    p_tags = p_tags/2
                    possible_tag = possible_tag/2"""
                    if use_emb :
                        possible_tag = emb_pred* tag_mul
                        p_tags = emb_pred
                
                """if num_tags == 12:
                    print(p_tags)"""
                #print(p_tags)
                # batch * 20 

                #p_tags = p_tags 
                
                # batch * 20 * seq_len
                p_starts = softmax(p_starts)
                p_ends = softmax(p_ends)
                #print(data['file_id'])
                # 20 tags
                for t in range(num_tags):
                    if middle and t != 2:
                        continue
                    pt = t
                    if num_tags == 12:
                        pt += 8

                    min_len = min_len_dict[pt]
                    max_len = max_len_dict[pt]
                    #if t == 5:
                        #print(p_tags[:,t])
                    # Candidate span
                    # Find the most possible one
                    if mode == 'dev':
                        tag_prob = tags[:,t] * p_tags[:,t]
                        dis_dict[pt] += tag_prob[tag_prob!=0]
                        """print(f"{t} : " )
                        print(tag_prob[tag_prob!=0])"""
                    for k,isTag in enumerate(p_tags[:,t]):
                        
                        key = data['file_id'][k]+'-'+str(data['index'][k].item())
                        if key not in prediction:
                            prediction[key] = {}
                        if data['index'][k].item() > 30:
                            if pt <8:
                                continue
                            isTag = possible_tag[k,t]
                        whole_sentence = self.tokenizer.decode(input_ids[k]).replace(" ","")
                        
                        
                        #print(data['index'][k].item())
                        # most 公告日(t == 7) in index 3, can be filtered in postprocessing
                        if isTag > threshold[pt] : #or (data['index'][k].item() == 3 and pt ==7):
                            
                            

                            #print(k,t)
                            s = -1
                            e = -1
                            max_p = 0
                            
                            pred_s, start = p_starts[k,t].topk(top)
                            pred_e, end = p_ends[k,t].topk(top)
                            for (ps,s_index) in zip(pred_s,start):
                                for (pe,e_index) in zip(pred_e,end):
                                    if s_index<=e_index :
                                        
                                        if max_p< ps.item()*pe.item() and e_index.item()-s_index.item()<50:
                                            max_p = ps.item()*pe.item()
                                            s = s_index.item()
                                            e = e_index.item()+1
                            if s >= e :
                                continue
                            
                            value = self.tokenizer.decode(input_ids[k,s:e])
                            pre_value = self.tokenizer.decode(input_ids[k,:s])
                            #self.filter(pt,value) and
                            if  max_len>len(value) >= min_len:
                                
                                #print(pre_value)
                                
                                pre_length  = len(pre_value)-5
                                if '<unk>' in pre_value :
                                    pre_length -=5
                                value = self.match_start_end(value,text[k],max(pre_length-1,0))
                                value = self.remove(pt,value)
                                if value == "":
                                    continue
                                #value = self.half2full(value)
                                if tag[pt] not in prediction[key]:
                                    prediction[key][tag[pt]] = ""
                                value = value.replace(" ","")
                                prediction[key][tag[pt]] += value
                        

                if i % 10 == 0:
                    print(f'{i+1}/{test_len}')
                elif (i+1) == test_len:
                    print(f'{i+1}/{test_len}')
        #print(dis_dict)
        return prediction

    def match_start_end(self,s,text,pre_length):
        s = s.replace(":"," ")
        if s[0] == " ":
            s = s[1:]
        s = self.full2half(s).lower()
        #print(pre_length)
        ch_text = self.full2half(text).lower()
        #print(ch_text)
        start = ch_text.find(s[0],pre_length)
        end = start + len(s)
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
        
        for s in ["から","まで"]:
            value = value.replace(s,"")
        """elif tag == 5 or tag == 6 or tag == 7 :
            encounters = ['日']
        elif tag == 9 or tag == 10 or tag == 11 or tag == 12:
            encounters = ['分']
        for enc in encounters:
            index = value.find(enc)
            if index >= 2 :
                break"""
        if index > 0:
            value = value[:index+1]
        return value
        