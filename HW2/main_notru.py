import torch
from preprocessing_notru import Preprocess
import json
from dataset import QADataset
from transformers import BertModel
import os
from model import Answer
from solver import Solver
import sys

arg = sys.argv
ctx_max_len=475
question_max = 30
pre = Preprocess(ctx_max_len=ctx_max_len,question_max=question_max)
data = {}
if not os.path.isdir('processed_data_notru'):
    os.mkdir('processed_data_notru')
if not os.path.isdir('ckpt'):
    os.mkdir('ckpt')
#
if arg[1] == 'train':
    for name in ['dev','train']:
        
        if not os.path.isfile(f'processed_data_notru/{name}.pkl'):
            print(f"Start {name}......")
            with open(f"data/{name}.json") as f:
                file = json.load(f)
                file = [data for data in file['data']]

            pre_data = pre.preprocess_data(file,train=not(name == 'test'),name=name)
            data[name] = QADataset(pre_data,padding=pre.tokenizer.pad_token_id,train=not(name == 'test'))
            torch.save(data[name],f'processed_data_notru/{name}.pkl')
        else:
            print(f"Load {name}......")
            data[name] = torch.load(f'processed_data_notru/{name}.pkl')
        data[name] = torch.utils.data.DataLoader(shuffle=False, dataset= data[name],batch_size=4 if name=='train' else 16,collate_fn=data[name].collate_fn)
        print(len(data[name]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert = BertModel.from_pretrained('bert-base-chinese').to(device)
    #print(bert)
    """for name,param in bert.named_parameters():
        param.requires_grad = False
        if "encoder.layer11" in name :
            break"""
      
    # +2 For CLS and SEP

    model = Answer(input_size=768,answer_size=pre.ctx_max_len+2).to(device)
    solver = Solver(device,pre.tokenizer)
    #print(bert)
    solver.train(data['train'],data['dev'],bert,model,ctx_max_len=pre.ctx_max_len+2)
elif arg[1] == 'test':
    for name in ['dev','test']:
        
        if not os.path.isfile(f'processed_data_notru/{name}_test.pkl'):
            print(f"Start {name}......")
            with open(f"data/{name}.json") as f:
                file = json.load(f)
                file = [data for data in file['data']]

            pre_data = pre.preprocess_data(file,train=not(name != 'train'))
            data[name] = QADataset(pre_data,padding=pre.tokenizer.pad_token_id,train=not(name != 'train'))
            torch.save(data[name],f'processed_data_notru/{name}_test.pkl')
        else:
            print(f"Load {name}......")
            data[name] = torch.load(f'processed_data_notru/{name}_test.pkl')
        data[name] = torch.utils.data.DataLoader(shuffle=False,dataset= data[name],batch_size=32,collate_fn=data[name].collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert = BertModel.from_pretrained('bert-base-chinese').to(device)
    #print(bert)
    for param in bert.parameters():
        param.requires_grad = False
    # +2 For CLS and SEP
    model = Answer(input_size=768,answer_size=pre.ctx_max_len+2).to(device)
    if os.path.isfile("ckpt/best_bert_notru.ckpt"):
        print("Load models........")
        model.linear_start.load_state_dict(torch.load("ckpt/best_linear_start_notru.ckpt",map_location= device))
        model.linear_end.load_state_dict(torch.load("ckpt/best_linear_end_notru.ckpt",map_location= device))
        model.linear_answerable.load_state_dict(torch.load("ckpt/best_class_notru.ckpt",map_location= device))
        bert.load_state_dict(torch.load("ckpt/best_bert_notru.ckpt",map_location= device))
        solver = Solver(device,pre.tokenizer)
        #print(bert)
        prediction = solver.test(data['dev'],bert,model,ctx_max_len=pre.ctx_max_len+2)
        file_name = arg[2]
        print(prediction)
        print("convert to json.......")
        with open(file_name,'w',encoding='utf8') as f :
            
            json.dump(prediction,f,ensure_ascii=False)
    else:
        print("No model CKPT file.")
else:
    file_name = arg[2]
    print(f"Start {name}......")
    with open(file_name) as f:
        file = json.load(f)
        file = [data for data in file['data']]

    pre_data = pre.preprocess_data(file,train=False)
    dataset = QADataset(pre_data,padding=pre.tokenizer.pad_token_id,train=not(name != 'train'))
    dataset = torch.utils.data.DataLoader(shuffle=False,dataset= dataset,batch_size=32,collate_fn=dataset.collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert = BertModel.from_pretrained('bert-base-chinese').to(device)
    for param in bert.parameters():
        param.requires_grad = False
    model = Answer(input_size=768,answer_size=pre.ctx_max_len+2).to(device)
    if os.path.isfile("best_bert.ckpt"):
        print("Load models........")
        model.linear_start.load_state_dict(torch.load("best_linear_start.ckpt",map_location= device))
        model.linear_end.load_state_dict(torch.load("best_linear_end.ckpt",map_location= device))
        model.linear_answerable.load_state_dict(torch.load("best_class.ckpt",map_location= device))
        bert.load_state_dict(torch.load("best_bert.ckpt",map_location= device))
        solver = Solver(device,pre.tokenizer)
        #print(bert)
        prediction = solver.test(dataset,bert,model,ctx_max_len=pre.ctx_max_len+2)
        #output file
        file_name = arg[3]
        print(prediction)
        print("convert to json.......")
        with open(file_name,'w',encoding='utf8') as f :
            
            json.dump(prediction,f,ensure_ascii=False)
    else:
        print("No model CKPT file.")