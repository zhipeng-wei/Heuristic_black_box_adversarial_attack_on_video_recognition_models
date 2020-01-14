import json
from torch.multiprocessing import Pool, Process, set_start_method
import torch
import os
import pickle
import argparse
import time
import random
import pandas as pd

def classify(model, inp):
    '''
    classify clips.
    return the top prob and class label.
    '''
    if inp.shape[0] != 1:
        inp = torch.unsqueeze(inp, 0)
    values, indices = torch.sort(-torch.nn.functional.softmax(model(inp)), dim=1)
    confidence_prob, pre_label = -float(values[:,0]), int(indices[:,0])
    return confidence_prob, pre_label

def get_attacked_targeted_label(model_name, data_name, attack_id):
    df = pd.read_csv('./attacked_samples-{}-{}.csv'.format(model_name, data_name))
    targeted_label = df[df['attack_id'] == attack_id]['targeted_label'].values.tolist()[0]
    return targeted_label

def get_attacked_samples(model, test_data, nums_attack, model_name, data_name):
    '''
    Generate idxs of test dataset for attacking.
    '''
    if os.path.exists('./attacked_samples-{}-{}.pkl'.format(model_name, data_name)):
        with open('./attacked_samples-{}-{}.pkl'.format(model_name, data_name), 'rb') as ipt:
            attacked_ids = pickle.load(ipt)
    else:
        random.seed(1024)
        idxs = random.sample(range(len(test_data)), len(test_data))
        attacked_ids = []
        for i in idxs:
            clips, label = test_data[i]
            video_id = label[0]
            label = int(label[1])
            _, pre = classify(model, clips)
            if pre != label:
                pass
            else:
                attacked_ids.append(i)
            if len(attacked_ids) == nums_attack:
                break
        with open('./attacked_samples-{}-{}.pkl'.format(model_name, data_name), 'wb') as opt:
            pickle.dump(attacked_ids, opt)
    return attacked_ids

def get_idx_labels(model, train_data, model_name, data_name):
    '''
    keys : ids in the training dataset.
    values: the label and the prediction of the model according to the ids.
    '''
    if os.path.exists('./train_dataset_ids_labels-{}-{}.pkl'.format(model_name, data_name)):
        with open('./train_dataset_ids_labels-{}-{}.pkl'.format(model_name, data_name), 'rb') as ipt:
            idx_to_labels = pickle.load(ipt)
    else:
        idx_to_labels = {}
        for i in range(len(train_data)):
            clips,label = train_data[i]
            _, pre = classify(model, clips)
            idx_to_labels[i] = [label, pre]
        with open('./train_dataset_ids_labels-{}-{}.pkl'.format(model_name, data_name), 'wb') as opt:
            pickle.dump(idx_to_labels, opt)
            
    return idx_to_labels

def get_initialize_samples(model, train_data, attack_id, num_samples, targeted, idx_to_labels):
    if targeted:
        ids = []
        for i in range(len(train_data)):
            label, pre = idx_to_labels[i]
            if pre == targeted:
                ids.append(i)       
        random.seed(attack_id)
        try:
            init_samples = sorted(random.sample(ids, num_samples))
        except:
            init_samples = ids
        return init_samples
    else:
        random.seed(attack_id)
        init_samples = sorted(random.sample(range(len(train_data)), num_samples))
        return init_samples
    
def untargetted_attack_one(model, train_data, attack_idx, x0, y0, init_samples, video_id, args):
    '''
    Attack one example.
    '''
    from untargeted_attack_utils import Attack_base
    output_path = args.output_path
    model_name = args.model_name
    del_frame, bound, bound_threshold, salient_region, spatial_mode, spatial_ratio, targeted, dataset_name = args.del_frame, args.bound, args.bound_threshold, args.salient_region, args.spatial_mode, args.spatial_ratio, args.targeted, args.dataset_name
    class_names = train_data.class_names
    label = y0
    one_class = Attack_base(model, train_data, attack_idx, x0, y0, output_path, model_name, dataset_name, init_samples, del_frame=del_frame, bound=bound, bound_threshold=bound_threshold, salient_region=salient_region, spatial_mode=spatial_mode, spatial_ratio=spatial_ratio)
    start = time.time()
    print ('Do attack_one {}'.format(attack_idx))
    one_class.attack()
    print ('Attack over {}'.format(attack_idx))
    end = time.time()
    print ('Success or not', one_class.success)
    if one_class.success:
        print ('Saving in path: {}.'.format(os.path.join(args.output_path, 'ori_adv_video_idx_{}'.format(attack_idx))))
        with open(os.path.join(args.output_path, 'ori_adv_video_idx_{}'.format(attack_idx)), 'wb') as opt_write:
            pickle.dump([one_class.image_ori, one_class.adv_image], opt_write)
        try:
            with open(os.path.join(args.output_path, 'MASK_video_idx_{}'.format(attack_idx)), 'wb') as opt_write:
                pickle.dump(one_class.best_MASK, opt_write)
        except AttributeError as e:
            pass
        with open(os.path.join(args.output_path, 'iteration_attack.log'), 'a') as the_file:
            json_ = json.dumps({
                'idx': attack_idx,
                'video_id':video_id,
                'targeted': targeted,
                'original_label': class_names[label],
                'original_confidence': one_class.ori_confi,
                'adversarial_label':  class_names[one_class.adv_indice],
                'adversarial_confidence': one_class.adv_confi,
                'query_counts': one_class.query_counts,
                'opt_counts': one_class.opt_counts, 
                'all_counts': one_class.query_counts + one_class.opt_counts,
                'P': '%.4f'%one_class.P.item(),
                'success': one_class.success,
                'time(mins)': (end-start)/60.0,
                'g2': one_class.g_theta.item(),
                'frame_indices': one_class.best_frame_indices,
                'additional': 'attack:Yes, success: True'
                })
            the_file.write(json_)
        
    else:
        with open(os.path.join(args.output_path, 'iteration_attack.log'), 'a') as the_file:
            json_ = json.dumps({
                    'idx': attack_idx,
                   'video_id':video_id,
                   'targeted': targeted,
                   'original_label': class_names[label],
                   'original_confidence': one_class.ori_confi,
                   'adversarial_label':  0,
                   'adversarial_confidence': 0,
                   'query_counts': one_class.query_counts,
                   'opt_counts':  one_class.opt_counts, 
                   'all_counts': one_class.query_counts + one_class.opt_counts,
                   'P': 0,
                   'success': False,
                   'time(mins)': (end-start)/60.0,
                   'g2': one_class.g_theta.item(),
                    'frame_indices': one_class.best_frame_indices,
                   'additional': 'attack: Yes, success: False'
                   })
            the_file.write(json_)
            
def targeted_attack_one(model, train_data, attack_idx, x0, y0, init_samples, video_id, args):
    '''
    Attack one example.
    '''
    from targeted_attack_utils import targeted_Attack_base
    output_path = args.output_path
    model_name = args.model_name
    del_frame, bound, bound_threshold, salient_region, spatial_mode, spatial_ratio, targeted, dataset_name = args.del_frame, args.bound, args.bound_threshold, args.salient_region, args.spatial_mode, args.spatial_ratio, args.targeted, args.dataset_name
    class_names = train_data.class_names
    label = y0
    one_class = targeted_Attack_base(model, train_data, attack_idx, x0, y0, targeted, output_path, model_name, dataset_name, init_samples, del_frame=del_frame, bound=bound, bound_threshold=bound_threshold, salient_region=salient_region, spatial_mode=spatial_mode, spatial_ratio=spatial_ratio)
    start = time.time()
    print ('Do attack_one {}'.format(attack_idx))
    one_class.attack()
    print ('Attack over {}'.format(attack_idx))
    end = time.time()
    print ('Success or not', one_class.success)
    if one_class.success:
        print ('Saving in path: {}.'.format(os.path.join(args.output_path, 'ori_adv_video_idx_{}'.format(attack_idx))))
        with open(os.path.join(args.output_path, 'ori_adv_video_idx_{}'.format(attack_idx)), 'wb') as opt_write:
            pickle.dump([one_class.image_ori, one_class.adv_image], opt_write)
        try:
            with open(os.path.join(args.output_path, 'MASK_video_idx_{}'.format(attack_idx)), 'wb') as opt_write:
                pickle.dump(one_class.best_MASK, opt_write)
        except AttributeError as e:
            pass
        with open(os.path.join(args.output_path, 'iteration_attack.log'), 'a') as the_file:
            json_ = json.dumps({
                'idx': attack_idx,
                'video_id':video_id,
                'targeted': targeted,
                'true_label': label,
                'original_label': class_names[label],
                'original_confidence': one_class.ori_confi,
                'adversarial_label':  class_names[one_class.adv_indice],
                'adversarial_confidence': one_class.adv_confi,
                'query_counts': one_class.query_counts,
                'opt_counts': one_class.opt_counts, 
                'all_counts': one_class.query_counts + one_class.opt_counts,
                'P': '%.4f'%one_class.P.item(),
                'success': one_class.success,
                'time(mins)': (end-start)/60.0,
                'g2': one_class.g_theta.item(),
                'frame_indices': one_class.best_frame_indices,
                'additional': 'attack:Yes, success: True'
                })
            the_file.write(json_)
        
    else:
        with open(os.path.join(args.output_path, 'iteration_attack.log'), 'a') as the_file:
            json_ = json.dumps({
                    'idx': attack_idx,
                   'video_id':video_id,
                   'targeted': targeted,
                   'true_label': label,
                   'original_label': class_names[label],
                   'original_confidence': one_class.ori_confi,
                   'adversarial_label':  0,
                   'adversarial_confidence': 0,
                   'query_counts': one_class.query_counts,
                   'opt_counts':  one_class.opt_counts, 
                   'all_counts': one_class.query_counts + one_class.opt_counts,
                   'P': 0,
                   'success': False,
                   'time(mins)': (end-start)/60.0,
                   'g2': one_class.g_theta,
                    'frame_indices': one_class.best_frame_indices,
                   'additional': 'attack: Yes, success: False'
                   })
            the_file.write(json_)
            
def save_parameters(args):
    def add_parameters(params, **kwargs):
        params.update(kwargs)
    params = {}
    add_parameters(params, data=args.dataset_name, model=args.model_name, nums_attack=args.nums_attack, path=args.path, n_process=args.n_process, del_frame=args.del_frame, bound=args.bound, salient_region=args.salient_region, bound_threshold=args.bound_threshold, spatial_mode=args.spatial_mode, spatial_ratio=args.spatial_ratio, output_path=args.output_path, target=args.target)
    json_ = json.dumps(params)
    with open(os.path.join(args.output_path, 'parameters.info'), 'w') as opt:
        opt.write(json_)