
import torch
import torch.nn as nn 
import torch.optim as optim
import random
import numpy as np
from model import TagValueModel,SentenceTaggingModel_2,Evaluation_Model
from preprocess import Preprocess
from dataset import TagValueDataset,DocumentDataset
from solver import Solver
from torch.optim.lr_scheduler import StepLR
import os
import json
import csv
seed = 9487
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
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
ver = "total_parent"
path = f"processed_data_{ver}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not os.path.isdir(path):
    os.mkdir(path)
if not os.path.isdir(f'ckpt_{ver}'):
    os.mkdir(f'ckpt_{ver}')
tokenizer = None
for mode in ['dev','train','test']:    
    if not os.path.isfile(f'{path}/{mode}.pkl'):
        dir_path = f'release/{mode}/ca_data'
        pre = Preprocess(dir_path,max_length=150,train=not(mode == 'test'))
        dataset,dataset_not_for_train = pre.process()
        data[mode] = TagValueDataset(dataset,tokenizer=pre.tokenizer,tags_num=pre.tags_num,train=not(mode == 'test'))
        data_not_train[mode] = dataset_not_for_train#TagValueDataset(dataset_not_for_train,tokenizer=pre.tokenizer,tags_num=pre.tags_num,train=not(mode == 'test'))
        torch.save(data[mode],f'{path}/{mode}.pkl')
        torch.save(data_not_train[mode],f'{path}/{mode}_not_train.pkl')
    else:
        print(f"Load {mode}......")
        data[mode] = torch.load(f'{path}/{mode}.pkl')
        data_not_train[mode] = torch.load(f'{path}/{mode}_not_train.pkl')
    
    tokenizer = data[mode].tokenizer
    #print(len(data[mode]))
    if mode != 'test':
        tags_num[mode] = data[mode].tags_num
        pos_weight[mode] = [((len(data[mode])-tag_num)/tag_num) for tag_num in tags_num[mode]]
        #print(data[mode][13])
    data[mode] = torch.utils.data.DataLoader(shuffle=False, dataset= data[mode],batch_size= 8)
    #data_not_train[mode] = torch.utils.data.DataLoader(shuffle=False, dataset= data_not_train[mode],batch_size= 1)
#print(data_not_train['dev'])
#print(pos_weight)
lr = 5e-6
model = TagValueModel().to(device)
optimizer = optim.AdamW(model.parameters(),lr=lr)
scheduler = StepLR(optimizer,1,gamma=0.9)
solver = Solver(device,tokenizer)
solver.train(data['train'],data['dev'],model,optimizer,pos_weight=pos_weight['train'],ver=ver)

tags_model = TagValueModel().to(device)
starts_model = TagValueModel().to(device)
ends_model = TagValueModel().to(device)
tags_model.load_state_dict(torch.load(f"ckpt_{ver}/tags.ckpt"))
ends_model.load_state_dict(torch.load(f"ckpt_{ver}/ends.ckpt"))
starts_model.load_state_dict(torch.load(f"ckpt_{ver}/starts.ckpt"))


cls = ""
print("##### Get Embedding #####")
for mode in ['dev','train','test']:
    if not os.path.isfile(f'{path}/emb_{mode}{cls}.pkl'):
        dir_path = f'release/{mode}/ca_data'
        pre = Preprocess(dir_path,max_length=180,train=not(mode == 'test'))
        
        embedding = solver.extract_sentence_embedding(data[mode],tags_model)
        
        #print(graph[0][list(graph[0].keys())[0]].size(),graph[1][list(graph[1].keys())[0]].size())
        data['e',mode] = DocumentDataset(embedding)
        
        torch.save(data['e',mode],f'{path}/emb_{mode}{cls}.pkl')
        
    else:
        print(f"Load {mode} Embedding......")
        data['e',mode] = torch.load(f'{path}/emb_{mode}{cls}.pkl')
        
    #print(data['fe',mode][3])
    
    data['e',mode] = torch.utils.data.DataLoader(shuffle=False, dataset= data['e',mode],batch_size= 1)
mode = 'dev'
name = 'gru_2'
class_lr = 2e-5
class_model = SentenceTaggingModel_2(attention=True).to(device)
optimizer = optim.AdamW(class_model.parameters(),lr=class_lr)
solver.two_stage_train(data['e','train'],data['e','dev'],class_model,optimizer,\
    pos_weight=pos_weight['train'],part='',ver=ver,gcn=False,name=name)
class_model.load_state_dict(torch.load(f"ckpt_{ver}/{name}_tags_new.ckpt"))
mode = 'dev'
threshold = [0.9]*20
Evaluation_model = Evaluation_Model(ckpt_cls=f"ckpt_{ver}/tags.ckpt"\
    ,ckpt_start=f"ckpt_{ver}/starts.ckpt",\
        ckpt_end=f"ckpt_{ver}/ends.ckpt").to(device)
#prediction = solver.test(data[mode],Evaluation_model,tag,test_emb=data['e',mode],two_stage_model=class_model,mode=mode,threshold=threshold,tag_top=20,use_emb=False)
prediction = solver.test(data[mode],Evaluation_model,tag,mode=mode,threshold=threshold)
pred = ""
appear_pdf = []
cur_index = 0
#print(prediction)
length = len(data_not_train[mode])
pre_index = 0
pre_key = ""
#print(length)
global_prediction = {}
for key, value in prediction.items():
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
        
    else:
        pred = pred[:-1] + '\n'
        out[-1] = ans[:-1]
    global_prediction[key] = out[-1]
while cur_index < length:
    file_id = data_not_train[mode][cur_index]['file_id']
    index = data_not_train[mode][cur_index]['index']
    key = f"{file_id}-{index}"
    global_prediction[key] = "NONE"
    cur_index+=1
keys = global_prediction.keys()
def get_pri(x):
    index = x[10:]
    index = (3-len(index))*"0" + index 
    return x[:10]+index
keys = sorted(keys,key=get_pri)
with open(f'pred_{mode}_{ver}_{name}.csv', 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['ID','Prediction'])
    for key in keys:
        """if key[:9] not in appear_pdf:
            appear_pdf += [key[:9]] 
            writer.writerow([f"{key[:9]}-1",'NONE'])
            writer.writerow([f"{key[:9]}-2",'NONE'])
            continue"""
        writer.writerow([f"{key}",global_prediction[key]])

