
import torch
import torch.nn as nn 
import torch.optim as optim
import random
import numpy as np
from model import TagValueModel, \
    Evaluation_Model,SentenceTaggingModel,Encoder_Decoder,\
        SentenceTaggingModel_2,Combine_GRU_FCN_Model,SentenceCNNTaggingModel,\
            Combine_GRU_CNN_Model,SentenceCNNGRUModel,SentenceFCNTaggingModel,\
                SentenceGCNModel,SelfAttention,TagSentenceModel,Combine_GRU_FCN_Attention_Model
from preprocess_grand_parent_sib import Preprocess
from dataset import TagValueDataset,DocumentDataset
from solver import Solver
from torch.optim.lr_scheduler import StepLR
import sys
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
ver = sys.argv[1]
path = f"processed_data_{ver}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not os.path.isdir(path):
    os.mkdir(path)
if not os.path.isdir(f'ckpt_{ver}'):
    os.mkdir(f'ckpt_{ver}')
tokenizer = None
for mode in ['dev','train','test']:    
    if not os.path.isfile(f'{path}/b_{mode}.pkl'):
        dir_path = f'release/{mode}/ca_data'
        #,parent=sys.argv[2]=='T',grand_parent=sys.argv[3]=='T'
        pre = Preprocess(dir_path,max_length=180,train=not(mode == 'test'))
        front_dataset,back_dataset,dataset_not_for_train = pre.process()
        data['f',mode] = TagValueDataset(front_dataset,tokenizer=pre.tokenizer,tags_num=pre.tags_num,train=not(mode == 'test'))
        data['b',mode] = TagValueDataset(back_dataset,tokenizer=pre.tokenizer,tags_num=pre.tags_num,train=not(mode == 'test'))

        data_not_train[mode] = dataset_not_for_train#TagValueDataset(dataset_not_for_train,tokenizer=pre.tokenizer,tags_num=pre.tags_num,train=not(mode == 'test'))
        torch.save(data['f',mode],f'{path}/f_{mode}.pkl')
        torch.save(data['b',mode],f'{path}/b_{mode}.pkl')
        torch.save(data_not_train[mode],f'{path}/{mode}_not_train.pkl')
    else:
        print(f"Load {mode}......")
        data['f',mode] = torch.load(f'{path}/f_{mode}.pkl')
        data['b',mode] = torch.load(f'{path}/b_{mode}.pkl')
        data_not_train[mode] = torch.load(f'{path}/{mode}_not_train.pkl')
    
    tokenizer = data['f',mode].tokenizer
    #print(len(data[mode]))
    if mode != 'test':
        tags_num[mode] = data['f',mode].tags_num
        pos_weight['f',mode] = [((len(data['f',mode])-tag_num)/tag_num) for tag_num in tags_num[mode][:8]]
        pos_weight['b',mode] = [((len(data['f',mode])-tag_num)/tag_num) for tag_num in tags_num[mode][8:]]
        
        #print(data['b',mode][4])
    data['f',mode] = torch.utils.data.DataLoader(shuffle=False, dataset= data['f',mode],batch_size= 8)
    data['b',mode] = torch.utils.data.DataLoader(shuffle=False, dataset= data['b',mode],batch_size= 8)
    #data_not_train[mode] = torch.utils.data.DataLoader(shuffle=False, dataset= data_not_train[mode],batch_size= 1)
#print(data_not_train['dev'])
#print(pos_weight)
lr = 5e-6
print("##### Train Front part #####")
front_model = TagValueModel(num_tags=8).to(device)
optimizer = optim.AdamW(front_model.parameters(),lr=lr)
scheduler = StepLR(optimizer,1,gamma=0.9)
solver = Solver(device,tokenizer)
#solver.train(data['f','train'],data['f','dev'],front_model,optimizer,pos_weight=pos_weight['f','train'],part='front_',ver=ver,only_class=False)
print("##### Train Back part #####")
end_model = TagValueModel(num_tags=12).to(device)
optimizer = optim.AdamW(end_model.parameters(),lr=lr)
scheduler = StepLR(optimizer,1,gamma=0.9)
solver = Solver(device,tokenizer)
#solver.train(data['b','train'],data['b','dev'],end_model,optimizer,pos_weight=pos_weight['b','train'],part='back_',ver=ver,only_class=False)


emb_model_f = TagValueModel(num_tags=8).to(device)
emb_model_b = TagValueModel(num_tags=12).to(device)
emb_model_f.load_state_dict(torch.load(f'ckpt_{ver}/front_tags.ckpt'))
emb_model_b.load_state_dict(torch.load(f'ckpt_{ver}/back_tags.ckpt'))
cls = ""
print("##### Get Embedding #####")
for mode in ['dev','train','test']:
    if not os.path.isfile(f'{path}/emb_b_{mode}{cls}.pkl'):
        dir_path = f'release/{mode}/ca_data'
        pre = Preprocess(dir_path,max_length=180,train=not(mode == 'test'))
        #graph = pre.make_graph()
        front_embedding = solver.extract_sentence_embedding(data['f',mode],emb_model_f)
        back_embedding = solver.extract_sentence_embedding(data['b',mode],emb_model_b)
        #print(graph[0][list(graph[0].keys())[0]].size(),graph[1][list(graph[1].keys())[0]].size())
        data['fe',mode] = DocumentDataset(front_embedding)
        data['be',mode] = DocumentDataset(back_embedding)
        torch.save(data['fe',mode],f'{path}/emb_f_{mode}{cls}.pkl')
        torch.save(data['be',mode],f'{path}/emb_b_{mode}{cls}.pkl')
    else:
        print(f"Load {mode} Embedding......")
        data['fe',mode] = torch.load(f'{path}/emb_f_{mode}{cls}.pkl')
        data['be',mode] = torch.load(f'{path}/emb_b_{mode}{cls}.pkl')
    #print(data['fe',mode][3])
    
    data['fe',mode] = torch.utils.data.DataLoader(shuffle=False, dataset= data['fe',mode],batch_size= 1)
    data['be',mode] = torch.utils.data.DataLoader(shuffle=False, dataset= data['be',mode],batch_size= 1)
# gru, cnn, fcn, gru_cnn, gru_fcn,gru_2,att_gru, att_fcn, sa
name=sys.argv[2]
class_lr = 2e-5
if name == "gru_fcn":  
    front_class_model = Combine_GRU_FCN_Model(num_tags=8,attention=False).to(device)
    back_class_model = Combine_GRU_FCN_Model(num_tags=12,attention=False).to(device)
elif name == "cnn":
    front_class_model = SentenceCNNTaggingModel(num_tags=8,attention=False).to(device)
    back_class_model = SentenceCNNTaggingModel(num_tags=12,attention=False).to(device)
elif name == "gru":
    front_class_model = SentenceTaggingModel(num_tags=8,attention=False).to(device)
    back_class_model = SentenceTaggingModel(num_tags=12,attention=False).to(device)
elif name == "fcn":
    front_class_model = SentenceFCNTaggingModel(num_tags=8,attention=False).to(device)
    back_class_model = SentenceFCNTaggingModel(num_tags=12,attention=False).to(device)
elif name == "att_gru":
    front_class_model = SentenceTaggingModel(num_tags=8,attention=True).to(device)
    back_class_model = SentenceTaggingModel(num_tags=12,attention=True).to(device)
elif name == "att_fcn":
    front_class_model = SentenceFCNTaggingModel(num_tags=8,attention=True).to(device)
    back_class_model = SentenceFCNTaggingModel(num_tags=12,attention=True).to(device)
elif name == "sa":
    front_class_model = SelfAttention(num_tags=8,attention=True).to(device)
    back_class_model = SelfAttention(num_tags=12,attention=True).to(device)
elif name =="gru_2":
    front_class_model = SentenceTaggingModel_2(num_tags=8,attention=True).to(device)
    back_class_model = SentenceTaggingModel_2(num_tags=12,attention=True).to(device)
elif name == "att_gru_fcn":
    front_class_model = Combine_GRU_FCN_Model(num_tags=8,attention=True).to(device)
    back_class_model = Combine_GRU_FCN_Model(num_tags=12,attention=True).to(device)
f_optimizer = optim.AdamW(front_class_model.parameters(),lr=class_lr)
solver.two_stage_train(data['fe','train'],data['fe','dev'],front_class_model,f_optimizer,\
    pos_weight=pos_weight['f','train'],part='front_',ver=ver,gcn=False,name=name)

b_optimizer = optim.AdamW(back_class_model.parameters(),lr=class_lr)#
solver.two_stage_train(data['be','train'],data['be','dev'],back_class_model,b_optimizer,\
    pos_weight=pos_weight['b','train'],part='back_',ver=ver,gcn=False,name=name)



##### Load Evaluation Models ####
#front_class_model.load_state_dict(torch.load(f"ckpt_{ver}/gru_front_tags_97861.ckpt"))
#back_class_model.load_state_dict(torch.load(f"ckpt_{ver}/gru_2_back_tags_97244.ckpt"))
#back_class_model.load_state_dict(torch.load(f"ckpt_{ver}/fcn_back_tags_97633.ckpt"))

front_class_model.load_state_dict(torch.load(f"ckpt_{ver}/{name}_front_tags_new.ckpt"))
back_class_model.load_state_dict(torch.load(f"ckpt_{ver}/{name}_back_tags_new.ckpt"))

#front_class_model.load_state_dict(torch.load(f"ckpt_{ver}/gru_front_tags_new.ckpt"))
#back_class_model.load_state_dict(torch.load(f"ckpt_{ver}/gru_2_back_tags_new.ckpt"))

front_Evaluation_model = Evaluation_Model(num_tags=8,ckpt_cls=f"ckpt_{ver}/front_tags.ckpt"\
    ,ckpt_start=f"ckpt_{ver}/front_starts.ckpt",\
        ckpt_end=f"ckpt_{ver}/front_ends.ckpt").to(device)


back_Evaluation_model = Evaluation_Model(num_tags=12,ckpt_cls=f"ckpt_{ver}/back_tags.ckpt"\
    ,ckpt_start=f"ckpt_{ver}/back_starts.ckpt",\
        ckpt_end=f"ckpt_{ver}/back_ends.ckpt").to(device)


mode = 'dev'

### Result threshold ###

# 95711 on public two three sep, wrong attid and typeid
#threshold = [0.5,0.5,0.8,0.9,0.8,0.5,0.5,0.5,0.6,0.9,0.9,0.9,0.9,0.8,0.7,0.9,0.9,0.9,0.9,0.9]
#95722 three three, wrong attid and typeid
#threshold = [0.5,0.5,0.8,0.9,0.8,0.5,0.5,0.5,0.6,0.9,0.8,0.9,0.8,0.8,0.7,0.8,0.9,0.8,0.8,0.9]

#96145 three three sib, end start model reverse, wrong attid and typeid
#threshold = [0.9]*6 +[0.9]*14
#96224 three three sib right model, wrong attid and typeid
#threshold = [0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.8,0.5,0.9,0.9,0.9,0.9,0.9]

#96699 three three sib right model right attid and typeid
#threshold = [0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9]

#89911 on dev set (two stage) #89932 on dev set (front two stage back one stage) #96848 two stage front one stage back 
#threshold = [0.9]*20

# 90249 on dev back top_tag_num=12 use_emb=False both two stage, 97022 on public 
# ckpt_sep_three_three_sib_little_batch_size
threshold = [0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.95,0.9,0.8,0.8,0.9,0.9,0.95,0.95,0.9]
#_new 90352 on dev without sqrt
#threshold = [0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.7,0.8,0.9,0.8,0.95,0.95,0.9]

#threshold = [0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.7,0.8,0.9,0.8,0.95,0.95,0.9]
threshold = [0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.8,0.8,0.9,0.9,0.9,0.9,0.9]
threshold = [0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.8,0.8,0.9,0.9,0.9,0.9,0.9]
"""threshold = [0.95]*20
threshold[2]=0.9
threshold[13]=0.85"""
#97244 bidir+singledir
"""threshold = [0.9]*8+[0.8]*12
threshold[11] = 0.9
threshold[13] = 0.7"""
# 89997 on dev set (two stage)
#threshold = [0.9]*20
threshold = [0.9]*8+[0.7]*12

#97633
"""threshold = [0.9**(1/2)]*8+[0.8**(1/2)]*12
threshold[16] = 0.7**(1/2)
threshold[11] = 0.9**(1/2)
threshold[13] = 0.7**(1/2)"""
threshold = [0.9**(1/2)]*8+[0.8**(1/2)]*12
threshold[16] = 0.7**(1/2)
threshold[11] = 0.9**(1/2)
threshold[13] = 0.5**(1/2)

threshold[19] = 0.5**(1/2)
threshold = [0.9] *20
threshold[13]= 0.8
"""threshold[2]= 0.8
threshold[13]= 0.8
threshold[14]=0.7
threshold[16]= 0.8"""
print(threshold)
if sys.argv[3] == 'T':
    prediction_front = solver.test(data['f',mode],front_Evaluation_model,tag,test_emb=data['fe',mode],two_stage_model=front_class_model,mode=mode,threshold=threshold,tag_top=8,use_emb=False)
    prediction_back = solver.test(data['b',mode],back_Evaluation_model,tag,test_emb=data['be',mode],two_stage_model=back_class_model,mode=mode,threshold=threshold,tag_top=12,use_emb=False)
else:
    prediction_front = solver.test(data['f',mode],front_Evaluation_model,tag,mode=mode,threshold=threshold)
    prediction_back = solver.test(data['b',mode],back_Evaluation_model,tag,mode=mode,threshold=threshold)

#print(prediction_front)
#print(prediction_back)
pred = ""
appear_pdf = []
cur_index = 0
#print(prediction)
length = len(data_not_train[mode])
pre_index = 0
pre_key = ""
#print(length)
global_prediction = {}

for key, value in prediction_front.items():
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

for key, value in prediction_back.items():
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
with open(f'pred_{mode}_{ver}_{name}_cls.csv', 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['ID','Prediction'])
    for key in keys:
        """if key[:9] not in appear_pdf:
            appear_pdf += [key[:9]] 
            writer.writerow([f"{key[:9]}-1",'NONE'])
            writer.writerow([f"{key[:9]}-2",'NONE'])
            continue"""
        writer.writerow([f"{key}",global_prediction[key]])

