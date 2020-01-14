import pickle
import random
import pandas as pd
from dataset_load import generate_dataset

FILES = ['attacked_samples-c3d-hmdb51.pkl','attacked_samples-c3d-ucf101.pkl','attacked_samples-flownet-hmdb51.pkl',
'attacked_samples-flownet-ucf101.pkl','attacked_samples-lrcn-hmdb51.pkl','attacked_samples-lrcn-ucf101.pkl']

def model_dataset_name(filename):
    name = filename.split('.')[0]
    dataset_name, model_name = name.split('-')[-1], name.split('-')[-2]
    return dataset_name, model_name

def generate_targeted_idxs(filename):
    attack_id_target_df = pd.DataFrame(columns = ['attack_id', 'true_label', 'targeted_label'])
    
    with open(filename, 'rb') as ipt:
        attack_ids = pickle.load(ipt)
    
    dataset_name, model_name = model_dataset_name(filename)
    train_dataset, test_dataset = generate_dataset(model_name, dataset_name)
    
    targeted_labels = []
    y0s = []
    for tmp_id in attack_ids:
        _, label = test_dataset[tmp_id]        
        video_id = label[0]
        y0 = int(label[1])
        if 'hmdb51' in filename:
            random.seed(tmp_id)
            targeted_id = random.sample([k for k in range(51) if k != y0], 1)[0]            
        elif 'ucf101' in filename:
            random.seed(tmp_id)
            targeted_id = random.sample([k for k in range(101) if k != y0], 1)[0]
        targeted_labels.append(targeted_id)
        y0s.append(y0)
        
    attack_id_target_df['attack_id'] = attack_ids
    attack_id_target_df['true_label'] = y0s
    attack_id_target_df['targeted_label'] = targeted_labels
    return attack_id_target_df

if __name__ == '__main__':
    for filename in FILES:
        output_name = name = filename.split('.')[0] + '.csv'
        tmp_df = generate_targeted_idxs(filename)
        tmp_df.to_csv(output_name, index=False)

