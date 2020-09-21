
import torch
import torch.nn as nn 
import torch.optim as optim
from model import TagValueModel
from preprocess_grand_parent import Preprocess as p3
from preprocess import Preprocess as p2
from dataset import TagValueDataset
from solver import Solver
from torch.optim.lr_scheduler import StepLR
import os
import json
import csv

tag = {
            0 : "調達年度",
            1 : "都道府県",
            2 : "入札件名",
            3 : "施設名",
            4 :"需要場所(住所)",
            5 : "調達開始日",
            6 : "調達終了日",
            7 : "公告日",
            8 :"仕様書交付期限",
            9 :"質問票締切日時",
            10:"資格申請締切日時",
            11:"入札書締切日時",
            12:"開札日時",
            13:"質問箇所所属/担当者",
            14:"質問箇所TEL/FAX",
            15:"資格申請送付先",
            16:"資格申請送付先部署/担当者名",
            17:"入札書送付先",
            18:"入札書送付先部署/担当者名",
            19:"開札場所",  
            }
data = {}
data_not_train = {}
tags_num = {}
pos_weight = {}
ver = "com_three"
ver2 = "com_two"
path = f"processed_data_{ver}"
path2 = f"processed_data_{ver2}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not os.path.isdir(path):
    os.mkdir(path)
if not os.path.isdir(f'ckpt_{ver}'):
    os.mkdir(f'ckpt_{ver}')
tokenizer = None
for p in [path,path2]:
    for mode in ['dev','train','test']:    
        if not os.path.isfile(f'{p}/{mode}.pkl'):
            dir_path = f'release/{mode}/ca_data'
            pre3 = p3(dir_path,max_length=150,train=not(mode == 'test'))
            dataset,dataset_not_for_train = pre.process()
            data[p,mode] = TagValueDataset(dataset,tokenizer=pre3.tokenizer,tags_num=pre3.tags_num,train=not(mode == 'test'))
            data_not_train[p,mode] = dataset_not_for_train#TagValueDataset(dataset_not_for_train,tokenizer=pre.tokenizer,tags_num=pre.tags_num,train=not(mode == 'test'))
            torch.save(data[p,mode],f'{p}/{mode}.pkl')
            torch.save(data_not_train[mode],f'{p}/{mode}_not_train.pkl')
        else:
            print(f"Load {mode}......")
            data[p,mode] = torch.load(f'{p}/{mode}.pkl')
            data_not_train[p,mode] = torch.load(f'{p}/{mode}_not_train.pkl')
    
        tokenizer = data[p,mode].tokenizer
        #print(len(data[mode]))
        if mode != 'test':
            tags_num[p,mode] = data[p,mode].tags_num
            pos_weight[p,mode] = [((len(data[p,mode])-tag_num)/tag_num) for tag_num in tags_num[p,mode]]
            #print(data[mode][13])
        data[p,mode] = torch.utils.data.DataLoader(shuffle=False, dataset= data[p,mode],batch_size= 32)
    #data_not_train[mode] = torch.utils.data.DataLoader(shuffle=False, dataset= data_not_train[mode],batch_size= 1)
#print(data_not_train['dev'])
#print(pos_weight)
lr = 5e-6
model = TagValueModel().to(device)
optimizer = optim.AdamW(model.parameters(),lr=lr)
scheduler = StepLR(optimizer,1,gamma=0.9)
solver = Solver(device,tokenizer)
#solver.train(data['train'],data['dev'],model,optimizer,pos_weight=pos_weight['train'])

tags_model = TagValueModel().to(device)
starts_model = TagValueModel().to(device)
ends_model = TagValueModel().to(device)
tags_model.load_state_dict(torch.load(f"ckpt_{ver}/tags.ckpt"))
ends_model.load_state_dict(torch.load(f"ckpt_{ver}/ends.ckpt"))
starts_model.load_state_dict(torch.load(f"ckpt_{ver}/starts.ckpt"))

tags_model2 = TagValueModel().to(device)
starts_model2 = TagValueModel().to(device)
ends_model2 = TagValueModel().to(device)
tags_model2.load_state_dict(torch.load(f"ckpt_{ver2}/tags.ckpt"))
ends_model2.load_state_dict(torch.load(f"ckpt_{ver2}/ends.ckpt"))
starts_model2.load_state_dict(torch.load(f"ckpt_{ver2}/starts.ckpt"))
mode = 'test'
threshold = [0.9,0.3,0.7,0.8,0.8,0.4,0.6,0.8,0.8,0.8,0.8,0.8,0.8,0.9,0.7,0.9,0.85,0.85,0.9,0.9]

threshold = [0.7]*20
#threshold[5] = 0.6
print(threshold)
prediction = solver.test(data[path,mode],data[path2,mode],tags_model,ends_model,starts_model,tags_model2,ends_model2,starts_model2,tag,mode=mode,threshold=threshold)
pred = ""
appear_pdf = []
cur_index = 0
#print(prediction)
length = len(data_not_train[path,mode])
pre_index = 0
pre_key = ""
#print(length)
with open(f'pred_{mode}.csv', 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['ID','Prediction'])
    for key, value in prediction.items():
        #print(pre_key,key)
        if key[:9] not in appear_pdf:
            if len(appear_pdf) > 0 :
                while True  and length > 0:
                    file_id = data_not_train[path,mode][cur_index]['file_id']
                    index = data_not_train[path,mode][cur_index]['index']
                    if file_id not in appear_pdf:
                        pre_key = f"{file_id}-{index}"
                        break
                    
                    writer.writerow([f"{file_id}-{index}",'NONE'])
                    cur_index+=1
            appear_pdf += [key[:9]] 
            writer.writerow([f"{key[:9]}-1",'NONE'])
            writer.writerow([f"{key[:9]}-2",'NONE'])
            pred += f"{key[:9]}-1,NONE\n"
            pred += f"{key[:9]}-2,NONE\n"
            pre_key = key
            continue
        if int(pre_key[10:])+4< int(key[10:]):
            pre_index = -1
            while True  and length > 0:
                
                file_id = data_not_train[path,mode][cur_index]['file_id']
                
                
                index = data_not_train[path,mode][cur_index]['index']
                if pre_index > 0 and index-pre_index>4:
                    pre_key = f"{file_id}-{index}"
                    break
                pre_index = index
                writer.writerow([f"{file_id}-{index}",'NONE'])
                cur_index+=1
        pred += key
        pred += ","
        out = [key,""]
        hasAnswer = False
        ans = ""
        for t in range(20):
            if tag[t] in value.keys():
                pred += f'{tag[t]}:{value[tag[t]]} '
                ans += f'{tag[t]}:{value[tag[t]]} '
                hasAnswer = True
        
        if not hasAnswer:
            pred += 'NONE\n'
            out[-1] = "NONE"
            writer.writerow(out)
        else:
            pred = pred[:-1] + '\n'
            out[-1] = ans[:-1]
            writer.writerow(out)
        pre_key = key
    while True and cur_index < length:
        
        file_id = data_not_train[path,mode][cur_index]['file_id']
        index = data_not_train[path,mode][cur_index]['index']
        writer.writerow([f"{file_id}-{index}",'NONE'])
        cur_index+=1

"""with open('pred.json', 'w', encoding='utf8') as outfile:
    json.dump(prediction, outfile, ensure_ascii=False)"""

