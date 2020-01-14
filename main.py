import json
from torch.multiprocessing import Pool, Process, set_start_method
import torch
import os
import pickle
import argparse
import time
import random
from fuc_utils import *
import sys

from dataset_load import generate_dataset
from models_load import generate_model


#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

if __name__ == "__main__":
    # parameters
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset_name', type = str, default = 'hmdb51', help='the name of dataset.')
    parse.add_argument('--model_name',type = str, default = 'flownet', help='the name of video classification model.')
    parse.add_argument('--nums_attack', type = int, default=100, help='the number of attacked example.')
    parse.add_argument('--path', type = str, default = '/home/jingjing/zhipeng/adv-attack-video/generate/distance_attack')
    parse.add_argument('--n_process', type = int, default = 1, help='the number of multiprocesser.')
    # optical parameters, in common line, '--xxxx' exists and return True.
    parse.add_argument('--target', type = str, default='False', help='targeted attack or untargeted attack.')
    parse.add_argument('--del_frame', type = str, default='False', help='if or not use del_frame func.')
    parse.add_argument('--bound', type = str, default='False', help='if or not use bound func.')
    parse.add_argument('--salient_region', type = str, default='False', help='if or not use salinet_region func.')
    
    parse.add_argument('--bound_threshold', type = int, default=3, help='the threshold is used in bound func.')
    parse.add_argument('--spatial_mode', type = int, default=1, help='different cv2 funcs are used in salient_region func.')
    parse.add_argument('--spatial_ratio', type = float, default=0.6, help='the ratio in salient_region.')
    
    #parse.add_argument('--attack_part', type=int, default=0, help='We divide the attack_samples into ten parts. 0~9')
    #parse.add_argument('--part_num', type=int, default=10, help='The numbers of samples in the one part.')
    parse.add_argument('--spe_id', type=int, default=838)
    args = parse.parse_args()
    
    
    args.target = eval(args.target)
    args.del_frame = eval(args.del_frame)
    args.bound = eval(args.bound)
    args.salient_region = eval(args.salient_region)
    
    if not os.path.exists(args.path):
        args.path = '/home/jingjing/zhipeng/adv-attack-video/generate/distance_attack'
    if args.bound and args.del_frame:
        if args.salient_region:
            args.output_path = os.path.join(args.path, 'spatial_experiment-target_{}-data_{}-model_{}-delframe_{}-bound_{}_{}-salientregion_{}_{}'.format(args.target,args.dataset_name, args.model_name, args.del_frame, args.bound, args.bound_threshold, args.salient_region, args.spatial_ratio))
        else:
            args.output_path = os.path.join(args.path, 'spatial_experiment-target_{}-data_{}-model_{}-delframe_{}-bound_{}_{}-salientregion_{}'.format(args.target,args.dataset_name, args.model_name, args.del_frame, args.bound, args.bound_threshold, args.salient_region))
    else:
        args.output_path = os.path.join(args.path, 'spatial_experiment-target_{}-data_{}-model_{}-delframe_{}-bound_{}-salientregion_{}'.format(args.target,args.dataset_name, args.model_name, args.del_frame, args.bound, args.salient_region))

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    
    # saving parameters
    save_parameters(args)
    
    # get model, train_dataset, test_dataset and attacked ids in test_data.
    train_data, test_data = generate_dataset(args.model_name, args.dataset_name)
    
    print ('load modeling')
    model = generate_model(args.model_name, args.dataset_name)
    
    #attacked_ids = get_attacked_samples(model, test_data, args.nums_attack, args.model_name, args.dataset_name)
    
    
	#attacked_ids = attacked_ids[args.attack_part*args.part_num : args.attack_part*args.part_num + args.part_num]
    #print (attacked_ids)
    if args.target:
        idx_to_labels = get_idx_labels(model, train_data, args.model_name, args.dataset_name)
    
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    #for attack_idx in attacked_ids:
    for attack_idx in [args.spe_id]:
        if 'ori_adv_video_idx_{}'.format(attack_idx) in os.listdir(args.output_path):
            print ('This file: {} already exists.'.format('ori_adv_video_idx_{}'.format(attack_idx)))
            continue
        
        x0, label = test_data[attack_idx]
        video_id = label[0]
        y0 = int(label[1])
        if args.target:
            args.targeted = get_attacked_targeted_label(args.model_name, args.dataset_name, attack_idx)
            #if args.dataset_name == 'ucf101':
            #    random.seed(attack_idx)
            #    args.targeted = random.sample([k for k in range(101) if k!=y0], 1)[0]
            #elif args.dataset_name == 'hmdb51':
            #    random.seed(attack_idx)
            #    args.targeted = random.sample([k for k in range(51) if k!=y0], 1)[0]
        else:
            args.targeted = None
            
        if args.target:
            init_samples = get_initialize_samples(model, train_data, attack_idx, 100, args.targeted, idx_to_labels)
        else:
            init_samples = get_initialize_samples(model, train_data, attack_idx, 100, args.targeted)

        if args.target:
            print ('Targeted Attack Begins')
            targeted_attack_one(model, train_data, attack_idx, x0, y0, init_samples, video_id, args)
        else:
            print ('Untargeted Attack Begins!')
            untargetted_attack_one, (model, train_data, attack_idx, x0, y0, init_samples, video_id, args)

