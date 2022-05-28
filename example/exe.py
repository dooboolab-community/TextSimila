from TextSimila.text_sim_reco import Text_sim_reco
from TextSimila import preprocess
import argparse
import yaml
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('yaml_name', type=str, help='yaml file name in "./config"')
parser.add_argument('file_name', type=str, help='data file name in "./data"') 
parser.add_argument('--saved', default=True, help='whether to saved the model') 
parser.add_argument('--pretrain_tok', '-tok', default=False, help='whether to use pre-trained model for tokenization') 
parser.add_argument('--pretrain_emb', '-emb', default=False, help='whether to use pre-trained model for embedding') 
args = parser.parse_args()


# Set hyperparameter from yaml file
with open(os.path.join('./config', args.yaml_name)) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

path = config['path']
params_vid = config['text_sim_reco']
params_tok = config['tokenize']
params_emb = config['embedding']


# Load data
try:
    with open(os.path.join(path,args.file_name+'.json'), 'r', encoding='UTF8') as f:
        data = json.load(f)
except:
    raise FileNotFoundError(f"There's no data file. Put json file in './data' folder")

Items = [i['Items'] for i in data]
related_to_Items = [i['related_to_Items'] for i in data]


# Train and predict
text_sim_reco = Text_sim_reco(
        Items=Items,
        related_to_Items = related_to_Items,
        **params_vid,
        saved=args.saved,
        pretrain_tok=args.pretrain_tok,
        pretrain_emb=args.pretrain_emb)

text_sim_reco.train()

pred = text_sim_reco.predict()


# Save the output as json file
output_path = './output'
if not os.path.isdir(output_path):
    os.mkdir(output_path)

num = params_vid['reco_Item_number']
file_name = f'Top{num}_prediction'
extension = 'json'
file_name = preprocess.getFileName(file_name, extension, output_path)

with open(os.path.join(output_path, file_name),'w', encoding="utf-8") as f:
    json.dump({},f)

for verbose, (item, rec_item) in enumerate(zip(text_sim_reco.Items, pred)):
    if verbose < 10:
        print(item)
        for idx, rec in enumerate(rec_item):
            print(f'{idx+1}: {rec}')
        print()
    with open(os.path.join(output_path, file_name),'r', encoding="utf-8") as load_f:
        output = json.load(load_f)
        row = {item: rec_item}
        output.update(row)
        with open(os.path.join(output_path, file_name),'w', encoding="utf-8") as f:
            json.dump(output, f, indent=2)