
import pandas as pd
import unicodedata
import re
import glob
from transformers import BertTokenizer,AlbertTokenizer,AlbertModel
import torch


class Preprocess:
    def __init__(self,path,max_length=256,train=True):
        self.train = train
        self.max_length = max_length
        self.path = path
        self.dic = {
                    "Page No":0,	
                    "Text":1, 	
                    "Index":2,
                    "Parent Index":3,
                    "Is Title":4,
                    "Is Table":5,
                    "Tag":6,
                    "Value":7
                    }
        self.tag = {
                    "調達年度":0,
                    "都道府県":1,
                    "入札件名":2,
                    "施設名":3,
                    "需要場所(住所)":4,
                    "調達開始日":5,
                    "調達終了日":6,
                    "公告日":7,
                    "仕様書交付期限":8,
                    "質問票締切日時":9,
                    "資格申請締切日時":10,
                    "入札書締切日時":11,
                    "開札日時":12,
                    "質問箇所所属/担当者":13,
                    "質問箇所TEL/FAX":14,
                    "資格申請送付先":15,
                    "資格申請送付先部署/担当者名":16,
                    "入札書送付先":17,
                    "入札書送付先部署/担当者名":18,
                    "開札場所":19,  
                    }
        self.tags_num = [0]*20
        self.data_num = 0
        
        # Pretrained : [ bert-base-japanese ]
        #self.tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese',do_lower_case=True)
        # For AlBert  : 'ALINEAR/albert-japanese-v2'
        self.tokenizer = AlbertTokenizer.from_pretrained("ALINEAR/albert-japanese-v2")
        self.A = {}
        self.A_front = {}
        self.A_back = {}
        #model = AlbertModel.from_pretrained("ALINEAR/albert-japanese-v2")
        #print(model)
    
    def load_files(self):
        
        return sorted(glob.glob(self.path+'/*'))
    def get_start_end(self,text,value):
        start = text.find(value)
        
        # +1 for CLS token
        if start == -1:
            return -1 , -1
        # tokenized japanese has "_" in first word , no idea
        #print(text)
        #print(self.tokenizer.tokenize(text[:start]))
        encoded_text = self.tokenizer.tokenize(text)
        before_start = self.tokenizer.tokenize(text[:start])
        before_span = len(before_start)
        if text[start-1] == ' ':
            before_span+=1
        start = max(before_span+1,2 )
        # -1 means last word of value
        
        #print(self.tokenizer.tokenize(text))
        #print(self.tokenizer.tokenize(value))
        # -2 because of removing the "_"
        ori_value = value
        value = self.tokenizer.tokenize(value)
        #print(value)
        l = len(value)
        
        if '▁' == value[0]:
            
            l -=1
        
        end = start + l -1
        if '▁' in encoded_text[0]:
            
            if before_span == 0 and  '▁' != encoded_text[0]:
                
                
                start -= 1
                end -=1
        
            
        
                
        
        
        
        return start, end 
    def make_graph(self):
        files = self.load_files()
        front_num = 0
        back_num = 0
        for f,file in enumerate(files):
            # ID is File Name   
            file_id = file.split('.')[0].split('/')[-1]
            # Read rows without the first row ( column name )
            data = pd.read_excel(file, skiprows = 0, sheet_name = 0)
            rows = data.iloc()
            len_rows = 0
            
            self.index2row = {}
            for i, row in enumerate(rows):
                len_rows+=1
            self.A[file_id]=torch.zeros((len_rows,len_rows))
            back_start = -1
            via_money = False
            first_has_parent = -1
            for i, row in enumerate(rows):
                text = row[self.dic['Text']]
                if not via_money and (text[:2] == "２ " or text[:2] ==  "２．" or text[:2] ==  "2．" or text[:2] == "２." or text[:2] ==  "2."):
                    self.A_front[front_num] = self.A[file_id][first_has_parent:i,first_has_parent:i]
                    front_num+=1
                    via_money = True
                if via_money and (text[:2] == "３ " or text[:2] ==  "３．" or text[:2] ==  "3．" or text[:2] == "３." or text[:2] ==  "3."):
                    via_money = False
                    back_start = i
                if not via_money and "入札保証金" in text:
                    if back_start == -1:
                        self.A_front[front_num] = self.A[file_id][first_has_parent:i,first_has_parent:i]
                        front_num+=1
                        break
                    self.A_back[back_num] = self.A[file_id][back_start:i,back_start:i]
                    back_num+=1
                    via_money = True
                    break
                self.index2row[row[self.dic['Index']]] = i
                none = row.isnull()
                
                if not none[self.dic['Parent Index']]:
                    j = self.index2row[row[self.dic['Parent Index']]]
                    if first_has_parent < 0 :
                        first_has_parent = i
                    self.A[file_id][i,j] =1
                    self.A[file_id][j,i] =1
                    for k in range(j+1,i):
                        if self.A[file_id][k,j] == 1:
                            self.A[file_id][k,i] =1 
                            self.A[file_id][i,k] =1
        return self.A_front,self.A_back
    def process(self):
        print("###########Load File#############")
        files = self.load_files()
        print("###########Start Preprocessing#############")
        num_file = len(files)
        max_leng = 0
        preprocessed_data = []
        preprocessed_data_2 = []
        preprocessed_not_for_train = []
        preprocessed_data_middle = []
        for f,file in enumerate(files):
            # ID is File Name   
            file_id = file.split('.')[0].split('/')[-1]
            # Read rows without the first row ( column name )
            data = pd.read_excel(file, skiprows = 0, sheet_name = 0)
            
            
            rows = data.iloc()
            concat_length = 0
            l = 0
            ori_text = {}
            title_data = {}
            text_data = {}
            tags_data = {}
            values_data = {}
            parent_dict = {}
            length_data = {}
            start_data = {}
            end_data = {}
            text_dic = {}
            is_train = {}
            is_second={}
            # 入札保証金
            via_money = False
            second = False
            brother = []
            #print(file_id)
            for row in rows:
                text = row[self.dic['Text']]
                if not via_money and (text[:2] == "２ " or text[:2] ==  "２．" or text[:2] ==  "2．" or text[:2] == "２." or text[:2] ==  "2."):
                    brother = []
                    #via_money = True
                    second = True
                if via_money and (text[:2] == "３ " or text[:2] ==  "３．" or text[:2] ==  "3．" or text[:2] == "３." or text[:2] ==  "3."):
                    via_money = False
                    
                if not via_money and "入札保証金" in text:
                    brother = []
                    second=False
                    via_money = True
                text = text.replace(" ","")
                text = text.replace("　","")
                ori_text[row[self.dic['Index']]] = text
                
                #text = unicodedata.normalize("NFKC",re.sub('＊|\*|\s+', '',text))
                encoded_text = self.tokenizer.tokenize(text)
                none = row.isnull()
                tags = row[self.dic['Tag']]
                page = row[self.dic['Page No']]
                
                
                #print(row)
                #print(none,tags)
                key = -1
                text_dic[row[self.dic['Index']]] = encoded_text
                
                if not none[self.dic['Is Title']]:
                    brother = []
                    title_data[row[self.dic['Index']]] = encoded_text
                    #length_data[row[self.dic['Index']]] = 0
                    
                key = row[self.dic['Index']]
                if none[self.dic['Parent Index']]:
                    preprocessed_not_for_train += [{
                        'file_id' : file_id,
                        'index' : key,
                    }]
                    continue
                    
                if not none[self.dic['Parent Index']]:
                    if row[self.dic['Parent Index']] not in title_data.keys():
                        title_data[row[self.dic['Parent Index']]] = text_dic[row[self.dic['Parent Index']]]
                    key = row[self.dic['Index']]
                    parent_dict[key] = title_data[row[self.dic['Parent Index']]]
                    tags_data[key] = []
                    start_data[key] = []
                    end_data[key] = []
                    values_data[key] = []
                    length_data[key] = len(encoded_text)
                    text_to_saved = encoded_text
                    #print(encoded_text)
                    #TODO finish concate sib 
                    if row[self.dic['Parent Index']] in parent_dict.keys():
                        #if (key - 1) in parent_dict.keys()
                        max_leng = max(max_leng, len(encoded_text)+ len(title_data[row[self.dic['Parent Index']]]) + len(parent_dict[row[self.dic['Parent Index']]]))
                        encoded_text = self.tokenizer.encode_plus(encoded_text,parent_dict[row[self.dic['Parent Index']]] +title_data[row[self.dic['Parent Index']]]+brother,max_length=self.max_length, pad_to_max_length=True, add_special_tokens=True, return_tensors='pt')
                    else:
                        max_leng = max(max_leng, len(encoded_text)+ len(title_data[row[self.dic['Parent Index']]]))
                        encoded_text = self.tokenizer.encode_plus(encoded_text,title_data[row[self.dic['Parent Index']]]+brother,max_length=self.max_length, pad_to_max_length=True, add_special_tokens=True, return_tensors='pt')
                    brother = [self.tokenizer.sep_token]+text_to_saved
                    text_data[key] = encoded_text
                is_train[key] = not via_money
                is_second[key] = second
                if self.train:
                    if not none[self.dic['Tag']] and not none[self.dic['Value']]: 
                        tags = unicodedata.normalize("NFKC",re.sub('＊|\*|\s+', '',tags))
                        tags = tags.split(';')
                        values = row[self.dic['Value']]
                        #values = unicodedata.normalize("NFKC",re.sub('＊|\*|\s+', '',values))
                        values = values.split(';')
                        v_num = len(values)
                        if len(tags) > 1:
                            for t, tag in enumerate(tags):
                                if v_num > 1:
                                    value = values[t]
                                    
                                else:
                                    value = values[0]
                                if not via_money:
                                    self.tags_num[self.tag[tag]] += 1
                                    self.data_num += 1
                                value = value.replace(" ","")
                                tags_data[key] += [self.tag[tag]]
                                values_data[key] += [value]
                                start, end = self.get_start_end(text,value)
                                start_data[key] += [start]
                                end_data[key] += [end]
                                
                                #print(tag, value)
                        else:
                            for value in values:
                                
                                tag = tags[0]
                                if not via_money:
                                    self.tags_num[self.tag[tag]] += 1
                                    self.data_num += 1
                                value = value.replace(" ","")
                                value = value.replace("　","")
                                tags_data[key] += [self.tag[tag]]
                                values_data[key] += [value]
                                start, end = self.get_start_end(text,value)
                                start_data[key] += [start ]
                                end_data[key] += [end ]
                                
                                #print(tag, value)
                    else:
                        tags_data[key] += [-1]
                        values_data[key] += []
                        start_data[key] += [-1 ]
                        end_data[key] += [-1]
                        
            # structure of dictionary
            #
            # text_data: { 
            #               idx1 : {
            #                       "input_ids", 
            #                       "token_type_ids", 
            #                       "attention_mask"
            #                       }
            #               idx2 : .....
            #                     ...
            #            }
            #tag_data: { 
            #               idx1 : [tag1, tag2, tag3....]
            #               idx2 : .....
            #            }
            #start_data: { 
            #               idx1 : [tag1_start, tag2_start, tag3_start....]
            #               idx2 : .....
            #            }
            #end_data: { 
            #               idx1 : [tag1_end, tag2_end, tag3_end....]
            #               idx2 : .....
            #            }
            part = 0
            prev_is_for_train = False
            for key in text_data.keys():
                input_ids = text_data[key]["input_ids"][0]
                token_type_ids = text_data[key]["token_type_ids"][0]
                attention_mask = text_data[key]["attention_mask"][0]
                train = is_train[key] 
                text = "".join(ori_text[key])
                if self.train:
                    tags = torch.tensor([0]*20)
                    
                    for tag in tags_data[key]:
                        if tag != -1 :
                            tags[tag] = 1
                    starts = torch.tensor([-1]*20)
                    for s,start in enumerate(start_data[key]):
                        starts[tags_data[key][s]] = start
                    ends = torch.tensor([-1]*20)
                    for s,end in enumerate(end_data[key]):
                        ends[tags_data[key][s]] = end
                softmax_mask = torch.tensor([-1e9]*self.max_length)
                # 2 not 1 due to some situation may crash ( may be tokenizer )
                # mask [SEP] on postprocessing
                softmax_mask[1:1+length_data[key]] = 0
                if train:
                    if not prev_is_for_train:
                        prev_is_for_train = True
                        part +=1
                    if not is_second[key]:
                        preprocessed_data += [{
                            'file_id' : file_id,
                            'index' : key,
                            'input_ids' : input_ids,
                            'token_type_ids' : token_type_ids,
                            'attention_mask' : attention_mask,
                            'softmax_mask' : softmax_mask,
                            'train' : train,
                            'text' : text
                        }]
                        if self.train:
                            preprocessed_data[-1]['tags'] = tags[:8]
                            preprocessed_data[-1]['starts'] = starts[:8]
                            preprocessed_data[-1]['ends'] = ends[:8]
                    else:
                        preprocessed_data_2 += [{
                            'file_id' : file_id,
                            'index' : key,
                            'input_ids' : input_ids,
                            'token_type_ids' : token_type_ids,
                            'attention_mask' : attention_mask,
                            'softmax_mask' : softmax_mask,
                            'train' : train,
                            'text' : text
                        }]
                        if self.train:
                            preprocessed_data_2[-1]['tags'] = tags[8:]
                            preprocessed_data_2[-1]['starts'] = starts[8:]
                            preprocessed_data_2[-1]['ends'] = ends[8:]
                    
                else:
                    if prev_is_for_train:
                        prev_is_for_train = False
                    
                    preprocessed_not_for_train += [{
                        'file_id' : file_id,
                        'index' : key,
                    }]
                    

                

            if f % 10 == 0:
                print(f'{f+1}/{num_file}',end='\r')
            elif f == num_file-1:
                print(f'{f+1}/{num_file}',end='\n')
            
        print(max_leng)    
        return preprocessed_data,preprocessed_data_2,preprocessed_not_for_train
            

            
            
