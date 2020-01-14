# In direction-based method, there are two stages:
# 1. initialize direction.
# 2. estimate gradient-direction.
# 3. update direction.
# ************************************************
# need to do: ADAM update, initialize random direction
import time
import random 
import numpy as np
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import sys
import pickle
import os
import csv
import torch_dct as dct

from base_utils import *
from salient_region_utils import *


class targeted_Attack_base(object):
    def __init__(self, model, train_dataset, attack_idx, x0, y0, targeted, output_path, model_name, dataset_name, init_samples, num_samples=100, alpha=2, beta=0.005, iterations=1000, d=20, q=20, del_frame=False, bound=False, bound_threshold=3, salient_region=False, spatial_mode=1, spatial_ratio=0.6):
        '''
        Params:
            model: pretrained predict model.
            train_dataset: used in model for training, used to initialize.
            attack_idx: the indice of attack image in test dataset.
            x0: attack image.
            y0: attack label.
            alpha: the step size with gradient to update direction.
            beta: used to estimate gradient.
            iterations: the max iteartions to update direction.
            num_samples: initialize from the number of train samples.
            b: the coefficient of estimated-gradient in AutoZOOM.
            q：calculate mean gradient by q different direction.
            output_path: the output dir path.
            model_name: the type of model.
            ADAM: if or not use ADAM update.
            mask: list, indices of mask.
            randn: if or not use different noise in estimated-gradient.
        '''
        # basic parameters
        self.model = model
        self.train_dataset = train_dataset
        self.attack_idx = attack_idx
        self.x0 = x0
        self.y0 = y0
        self.targeted = targeted
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.num_samples = num_samples
        self.d = d
        self.q = q
        self.output_path = output_path
        self.model_name = model_name
        self.dataset_name = dataset_name
        
        self.samples = init_samples
        #optical parameters
        self.del_frame = del_frame
        self.bound = bound
        self.salient_region = salient_region
        self.bound_threshold = bound_threshold
        self.spatial_ratio = spatial_ratio
        self.spatial_mode = spatial_mode
        
        self.query_counts = 0
        self.opt_counts = 0
        try:
            self.model.cuda()
        except:
            pass
        self.x0 = self.x0.cuda()
        self.ori_confi, self.ori_label = self.classify(self.x0)
        self.initialize_paras()
        
    def initialize_paras(self):
        '''initialize parameters'''
        if self.model_name == 'c3d':
            input_type = 'each pixel value sub mean, range(-1xx, +1xx) to (-0.x, +0.x)'
            DIV = 255
            input_size = '(batch_size, num_channels, seq_len, height, width)'
            self.seq_len = self.x0.size()[1]
            self.seq_axis = 1
        elif self.model_name == 'lrcn':
            input_type = 'each pixel value div 255, range(0, 1) to (0, 1)'
            DIV = 1
            input_size = '(batch_size, seq_len, height, width, num_channels)'
            self.seq_len = self.x0.size()[0]
            self.seq_axis = 0
        elif self.model_name == 'flownet':
            input_type = 'each pixel value remain unchanged, range(0, 255) to (0, 1).'
            DIV = 255
            input_size = '(batch_size, seq_len, num_channels, height, width)'
            self.seq_len = self.x0.size()[0]
            self.seq_axis = 0
            
        # 如果都是同一个random_seed，那么对于不同的被攻击样本的初始化样本总是一样的，因此使用attack_idx作为random_seed.        
        self.image_ori = vector_to_image(self.model_name, self.dataset_name, self.x0)
        self.MASK = torch.ones(self.x0.size()).cuda()
        
    def classify(self, inp, state=None):
        if inp.shape[0] != 1:
            inp = torch.unsqueeze(inp, 0)
        values, indices = torch.sort(-torch.nn.functional.softmax(self.model(inp)), dim=1)
        confidence_prob, pre_label = -float(values[:,0]), int(indices[:,0])
        if state == 'query':
            self.query_counts += 1
        elif state == 'opt':
            self.opt_counts += 1
        return confidence_prob, pre_label
    
    def initialize_salient_region_mask(self):
        cv2_func = get_cv2_func(self.spatial_mode)
        MASKs = []
        for i in range(self.seq_len):
            if self.seq_axis == 1:
                tmp_image = self.image_ori[:,i,:,:]
            else:
                tmp_image = self.image_ori[i]
            this_mask = SpectralResidual(cv2_func, tmp_image.cpu(), self.spatial_ratio, self.model_name)
            MASKs.append(this_mask)
            
        
        # [seq_len, height, width, num_channels]
        if self.model_name == 'c3d':
            MASKs = torch.stack(MASKs).permute(3,0,1,2).cuda()
        elif self.model_name == 'lrcn':
            MASKs = torch.stack(MASKs).cuda()
        elif self.model_name == 'flownet':
            MASKs = torch.stack(MASKs).permute(0,3,1,2).cuda()
        self.spatial_MASK = MASKs

        
    def frames_to_mask(self, frame_indices):
        mask = torch.zeros(self.x0.size())
        if self.model_name == 'c3d':
            mask[:,frame_indices,:,:] = 1
        else:
            mask[frame_indices,:,:,:] = 1
        return mask.cuda() * self.MASK
    
    def get_bounding_value(self, frame_indices, noise_vector, image_adv):
        '''平均扰动要与tmp_p进行对比，如果小于，减少帧，如果大于，减少扰动
            知道该初始方向上的最佳对抗样本。'''
        bound_mask = self.frames_to_mask(frame_indices)
        DIV = 255
        tmp_vector = noise_vector * bound_mask + self.x0 
        #image_to_vector(self.model_name, (image_adv-self.image_ori)*bound_mask+self.image_ori)
        this_prob, this_pre = self.classify(tmp_vector, 'query')
        
        if this_pre == self.targeted:
            theta = (image_adv-self.image_ori)/DIV * bound_mask
            initial_lbd = torch.norm(theta)
            theta = theta/initial_lbd
            binary_search_start = time.time()
            lbd = self.fine_grained_binary_search(theta, initial_lbd, bound_mask)
            if lbd == float('inf'):
                return None
            tmp_image_noise = theta * lbd * DIV * bound_mask
            
            all_nums = int(torch.sum(bound_mask.reshape(-1)).item())
            valid_indices = torch.argsort(-bound_mask.reshape(-1))[:all_nums]
            tmp_p = torch.mean(torch.abs(tmp_image_noise.reshape(-1)[valid_indices]))
            return (frame_indices, initial_lbd, lbd, theta, binary_search_start, tmp_p, this_pre)
        else:
            return None
    
    def spatial_sort_frames(self, noise_vector):
        all_frames = [i for i in range(16)]
        score_dict = {}
        for i in all_frames:
            tmp_MASK = torch.ones(self.x0.size()).cuda()
            if self.model_name == 'c3d':
                tmp_MASK[:, i, :, :] = self.spatial_MASK[:, i, :, :]
            else:
                tmp_MASK[i, :, :, :] = self.spatial_MASK[i, :, :, :]
            tmp_adv = noise_vector * tmp_MASK + self.x0
            this_prob, this_pre = self.classify(tmp_adv, 'query')
            if this_pre == self.targeted:
                score_dict[i] = this_prob
            else:
                pass
        if score_dict:
            sorted_items = sorted(score_dict.items(), key=lambda x:-x[1])
            return sorted_items
        else:
            return None
        
    def loop_del_frame_with_spatial(self, noise_vector):
        sorted_items = self.spatial_sort_frames(noise_vector)
        #print ('sorted_items in spatial', sorted_items)
        one_mask = torch.ones(self.x0.size()).cuda()
        return_mask = torch.ones(self.x0.size()).cuda()
        print ('ratio', torch.sum(self.spatial_MASK)/torch.sum(return_mask))
        if sorted_items:
            sorted_frames = []
            for item in sorted_items:
                sorted_frames.append(item[0])
            spatial_frame = []
            del_flag = 0
            for this_frame in sorted_frames:
                print (torch.sum(return_mask)/torch.sum(one_mask))
                if self.model_name == 'c3d':
                    return_mask[:, this_frame, :, :] = self.spatial_MASK[:, this_frame, :, :]
                else:
                    return_mask[this_frame, :, :, :] = self.spatial_MASK[this_frame, :, :, :]                
                tmp_adv = noise_vector * return_mask + self.x0
                this_prob, this_pre = self.classify(tmp_adv, 'query')
                #print ('iteration-{}, pre-{}, targeted-{}'.format(this_frame, this_pre, self.targeted))
                if this_pre == self.targeted:
                    spatial_frame.append(this_frame)
                    del_flag += 1
                else:
                    if self.model_name == 'c3d':
                        return_mask[:, this_frame, :, :] = one_mask[:, this_frame, :, :]
                    else:
                        return_mask[this_frame, :, :, :] = one_mask[this_frame, :, :, :]                    
            print ('{} frames are spatial!'.format(del_flag))
            self.MASK = return_mask
        else:
            self.MASK = torch.ones(self.x0.size()).cuda()
        
    def loop_del_frame_sort_sequence(self, noise_vector, mode='target'):
        all_frames = [i for i in range(16)]
        score_dict = {}
        for i in all_frames:
            tmp_frames = [_ for _ in all_frames if _!=i]
            tmp_mask = self.frames_to_mask(tmp_frames)
            tmp_vector_adv = noise_vector * tmp_mask + self.x0
            this_prob, this_pre = self.classify(tmp_vector_adv, 'query')
            if mode == 'target':
                if this_pre != self.targeted:
                    pass
                else:
                    score_dict[i] = this_prob
            elif mode == 'untarget':
                if this_pre != self.y0:
                    score_dict[i] = this_prob
                else:
                    pass
        if score_dict:
            sorted_items = sorted(score_dict.items(), key=lambda x:-x[1])
            sorted_frames = []
            for item in sorted_items:
                sorted_frames.append(item[0])
            return sorted_frames
        else:
            return None
    
    def loop_del_frame(self, noise_vector, mode='target'):
        sorted_frames = self.loop_del_frame_sort_sequence(noise_vector, mode)
        all_frames = [i for i in range(16)]
        if sorted_frames:
            for i in sorted_frames:
                tmp_frames = [k for k in all_frames if k!=i]
                tmp_mask = self.frames_to_mask(tmp_frames)
                tmp_vector_adv = noise_vector * tmp_mask + self.x0
                this_prob, this_pre = self.classify(tmp_vector_adv, 'query')
                if mode == 'untarget':
                    if this_pre != self.y0:
                        all_frames = tmp_frames
                        continue
                    else:
                        pass
                elif mode == 'target':
                    if this_pre == self.targeted:
                        all_frames = tmp_frames
                        continue
                    else:
                        pass
            
        return all_frames
            
    
    def initialize_from_train_dataset_del_frame_bound(self):
        '''
        Initialize theta(direction) and g2(distance along with the direction)
        '''
        attack_initialize_logger = Logger(
                os.path.join(self.output_path, 'attack_initialize_from_train_{}.log'.format(self.attack_idx)),
                ['process', 'attack_idx', 'initial_idx', 'ori_lbd', 'cur_lbd', 'counts', 'this_label', 'ori_label', 'P', 'masked_frames', 'search_time(mins)'])
        
        outer_best_p = float('inf')
        best_theta, g_theta = None, float('inf')
        self.best_MASK = None
        self.best_frame_indices = None

        
        for i in self.samples:
            xi, yi = self.train_dataset[i]
            xi = xi.cuda()
            
            noise_vector = xi-self.x0
            if self.salient_region:
                self.loop_del_frame_with_spatial(noise_vector)
                vector_adv = noise_vector * self.MASK + self.x0
                this_prob, this_pre = self.classify(vector_adv, 'query')
                print ('salient, pre-true', this_pre, self.targeted)
            else:
                vector_adv = noise_vector * self.MASK + self.x0
                this_prob, this_pre = self.classify(vector_adv, 'query')
            if this_pre == self.targeted:
                image_adv = vector_to_image(self.model_name, self.dataset_name, vector_adv)
                del_frame_sequences = self.loop_del_frame_sort_sequence(noise_vector)
                print ('this del_frame_sequence', del_frame_sequences)
                #del_frame_sequences = loop_del_frame_sort_sequence(image_adv.cpu(), self.image_ori.cpu(), self.targeted, self.model, self.model_name, tmp_MASK, 'target')
                if not del_frame_sequences:
                    continue
                    
                begin_frames = [k for k in range(self.seq_len)]
                re = self.get_bounding_value(begin_frames, noise_vector, image_adv)
                
                if not re:
                    continue
                    
                frame_indices, initial_lbd, lbd, theta, binary_search_start, tmp_p, this_pre = re
                
                # 定义变量进行筛选
                inner_frames = frame_indices
                inner_p = tmp_p
                inner_lbd = lbd
                inner_theta = theta
                
                attack_initialize_logger.log({
                    'process': 'initialize_theta_search_best_p',
                    'attack_idx': '%4d'%self.attack_idx,
                    'initial_idx': '%4d'%i,
                    'ori_lbd': '%.4f'%initial_lbd,
                    'cur_lbd': '%.4f'%inner_lbd,
                    'counts': '%4d'%self.query_counts,
                    'this_label': '%4d'%this_pre,
                    'ori_label': '%4d'%self.y0,
                    'P' : inner_p,
                    'masked_frames': '-'.join([str(i) for i in inner_frames]),
                    'search_time(mins)': (time.time()-binary_search_start)/60.0
                    })
                
                for del_frame in del_frame_sequences:
                    print ('del frame, cycle', del_frame)
                    tmp_frames = [i for i in inner_frames if i != del_frame]                    
                    re = self.get_bounding_value(tmp_frames, noise_vector, image_adv)
                    if re:
                        frame_indices, initial_lbd, lbd, theta, binary_search_start, tmp_p, this_pre = re
                        
                        
                        
                        # 如果大于bound_threshold，那么按照减少扰动的方向移动
                        #print ('inner_p', inner_p, self.bound_threshold, tmp_p)
                        if inner_p >= self.bound_threshold:
                            if tmp_p < inner_p:
                                inner_frames = tmp_frames
                                inner_p = tmp_p
                                inner_lbd = lbd
                                inner_theta = theta
                        # 如果小于bound_threshod，那么按照较少帧数的方向移动
                        else:
                            if len(tmp_frames) < len(inner_frames):
                                inner_frames = tmp_frames
                                inner_p = tmp_p
                                inner_lbd = lbd
                                inner_theta = theta
                                
                   
                        
                        attack_initialize_logger.log({
                                'process': 'initialize_theta_search_best_p',
                                'attack_idx': '%4d'%self.attack_idx,
                                'initial_idx': '%4d'%i,
                                'ori_lbd': '%.4f'%initial_lbd,
                                'cur_lbd': '%.4f'%inner_lbd,
                                'counts': '%4d'%self.query_counts,
                                'this_label': '%4d'%this_pre,
                                'ori_label': '%4d'%self.y0,
                                'P' : inner_p,
                                'masked_frames': '-'.join([str(i) for i in inner_frames]),
                                'search_time(mins)': (time.time()-binary_search_start)/60.0
                                })

                    else:
                        
                        continue
                
                attack_initialize_logger.log({
                        'process': 'initialize_theta',
                        'attack_idx': '%4d'%self.attack_idx,
                        'initial_idx': '%4d'%i,
                        'ori_lbd': '%.4f'%10086,
                        'cur_lbd': '%.4f'%inner_lbd,
                        'counts': '%4d'%self.query_counts,
                        'this_label': '%4d'%self.targeted,
                        'ori_label': '%4d'%self.y0,
                        'P' : '%4d'%inner_p,
                        'masked_frames': '-'.join([str(i) for i in inner_frames]),
                        'search_time(mins)': (time.time()-binary_search_start)/60.0
                        })

                
                if inner_p < outer_best_p:
                    best_theta, g_theta = inner_theta, inner_lbd
                    outer_best_p = inner_p
                    self.best_frame_indices = inner_frames 
                    best_mask = self.frames_to_mask(inner_frames)

                
        self.best_MASK = best_mask
        self.MASK = best_mask
        self.best_theta, self.g_theta = best_theta, g_theta
        self.theta, self.g2 = best_theta, g_theta        
    
    def initialize_from_train_dataset_del_frame(self):
        '''
        Initialize theta(direction) and g2(distance along with the direction)
        '''
        attack_initialize_logger = Logger(
                os.path.join(self.output_path, 'attack_initialize_from_train_{}.log'.format(self.attack_idx)),
                ['process', 'attack_idx', 'initial_idx', 'ori_lbd', 'cur_lbd', 'counts', 'this_label', 'ori_label', 'masked_frames', 'search_time(mins)'])
        
        best_theta, g_theta = None, float('inf')
        self.best_MASK = None
        self.best_frame_indices = None
        
        for i in self.samples:
            xi, yi = self.train_dataset[i]
            xi = xi.cuda()
            noise_vector = xi-self.x0
            if self.salient_region:
                self.loop_del_frame_with_spatial(noise_vector)
                vector_adv = noise_vector * self.MASK + self.x0
                this_prob, this_pre = self.classify(vector_adv, 'query')
            else:
                vector_adv = noise_vector * self.MASK + self.x0
                this_prob, this_pre = self.classify(vector_adv, 'query')
                

            if this_pre == self.targeted:
                image_adv = vector_to_image(self.model_name, self.dataset_name, vector_adv)
                frame_indx = self.loop_del_frame(noise_vector)
                tmp_MASK = self.frames_to_mask(frame_indx)
                
                #del_flag, MASK, frame_indices, this_query  = loop_del_frame_sort(image_adv.cpu(), self.image_ori.cpu(), self.targeted, self.model, self.model_name, tmp_MASK, 'target')
                
                DIV = 255
                theta = (image_adv-self.image_ori)/DIV * tmp_MASK
                initial_lbd = torch.norm(theta)
                theta = theta/initial_lbd
                binary_search_start = time.time()
                
                #vector_adv_tmp = image_to_vector(self.model_name, theta*initial_lbd*DIV + self.image_ori)
                #this_prob, this_pre = self.classify(vector_adv_tmp, 'query')
                #print ('Number2, pre-target{}-{}'.format(this_pre, self.targeted))
                
                #if this_pre != self.targeted:
                #    break
                
                lbd = self.fine_grained_binary_search(theta, initial_lbd,tmp_MASK)
                attack_initialize_logger.log({
                    'process': 'initialize_theta',
                    'attack_idx': '%4d'%self.attack_idx,
                    'initial_idx': '%4d'%i,
                    'ori_lbd': '%.4f'%initial_lbd,
                    'cur_lbd': '%.4f'%lbd,
                    'counts': '%4d'%self.query_counts,
                    'this_label': '%4d'%self.targeted,
                    'ori_label': '%4d'%self.y0,
                    'masked_frames': '-'.join([str(i) for i in frame_indx]),
                    'search_time(mins)': (time.time()-binary_search_start)/60.0
                    })
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    self.best_frame_indices = frame_indx
                    best_mask = self.frames_to_mask(frame_indx)
                    
        self.best_MASK = best_mask
        self.MASK = best_mask
        self.best_theta, self.g_theta = best_theta, g_theta
        self.theta, self.g2 = best_theta, g_theta
     
    
    def initialize_from_train_dataset_baseline(self):
        '''
        Initialize theta(direction) and g2(distance along with the direction)
        '''
        attack_initialize_logger = Logger(
                os.path.join(self.output_path, 'attack_initialize_from_train_{}.log'.format(self.attack_idx)),
                ['process', 'attack_idx', 'initial_idx', 'ori_lbd', 'cur_lbd', 'counts', 'this_label', 'ori_label', 'masked_frames', 'search_time(mins)'])
        
        best_theta, g_theta = None, float('inf')
        self.best_MASK = None
        self.best_frame_indices = None
        for i in self.samples:
            xi, yi = self.train_dataset[i]
            xi = xi.cuda()
            this_prob, this_pre = self.classify(xi, 'query')
            if this_pre == self.targeted:
                image_adv = vector_to_image(self.model_name, self.dataset_name, xi)
                DIV = 255
                theta = (image_adv-self.image_ori)/DIV * self.MASK
                initial_lbd = torch.norm(theta)
                theta = theta/initial_lbd
                binary_search_start = time.time()
                lbd = self.fine_grained_binary_search(theta, initial_lbd, self.MASK)
                attack_initialize_logger.log({
                    'process': 'initialize_theta',
                    'attack_idx': '%4d'%self.attack_idx,
                    'initial_idx': '%4d'%i,
                    'ori_lbd': '%.4f'%initial_lbd,
                    'cur_lbd': '%.4f'%lbd,
                    'counts': '%4d'%self.query_counts,
                    'this_label': '%4d'%self.targeted,
                    'ori_label': '%4d'%self.y0,
                    'masked_frames': 'all',
                    'search_time(mins)': (time.time()-binary_search_start)/60.0
                    })
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    
        self.best_theta, self.g_theta = best_theta, g_theta
        self.theta, self.g2 = best_theta, g_theta
        self.best_MASK = self.MASK
    
    
    def fine_grained_binary_search(self, theta, initial_lbd, this_mask):
        lbd = initial_lbd
        while self.classify(image_to_vector(self.model_name, self.dataset_name, self.image_ori+lbd*this_mask*theta*255), 'query')[1] != self.targeted:
            lbd *= 1.05
            if lbd > 300:
                return float('inf')

        num_intervals = 100
        lambdas = np.linspace(0.0, lbd.cpu(), num_intervals)[1:]
        lbd_hi = lbd
        lbd_hi_index = 0
        for i, lbd in enumerate(lambdas):
            if self.classify(image_to_vector(self.model_name, self.dataset_name, self.image_ori+lbd*this_mask*theta*255), 'query')[1] == self.targeted:
                lbd_hi = lbd
                lbd_hi_index = i
                break
        lbd_lo = lambdas[lbd_hi_index-1]
        while (lbd_hi - lbd_lo) > 1e-7:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            if self.classify(image_to_vector(self.model_name, self.dataset_name, self.image_ori+lbd_mid*this_mask*theta*255), 'query')[1] == self.targeted:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
                
        return lbd_hi


    def estimate_gradient(self, iteration_indice):
        gradient = torch.zeros(self.x0.size()).cuda()
        self.min_g1_theta, self.min_g1 = None, float('inf')
        gradient_flag = 0
        valid_flag = 0
        for i in range(self.q):
            gradient_start = time.time()
            torch.manual_seed((iteration_indice+1)*(i+1))
            u = torch.randn(self.theta.shape, dtype = self.theta.dtype).cuda()
            u = u/torch.norm(u)
            new_theta = (self.theta + self.beta * u) * self.MASK
            new_theta = new_theta/torch.norm(new_theta)
            g1 = self.fine_grained_binary_search_local(new_theta, self.g2)
            
            if g1 == float('inf'):
                gradient_flag+=1
                continue
            # gradient += self.d * (g1-self.g2)/self.beta * u
            gradient += (g1-self.g2)/self.beta * u
            valid_flag+=1
            self.attack_gradient_logger.log({
                'process': 'calculate_gradient',
                'iteration_idx': '%4d'%iteration_indice, 
                'idx': '{}-{}'.format(i, self.q),
                'g1': '%.4f'%g1,
                'g2': '%.4f'%self.g2,
                'beta': '%.4f'%self.beta,
                'counts': '%4d'%self.opt_counts,
                'search_time(mins)':(time.time()-gradient_start)/60.0
                })
            if g1 < self.min_g1:
                self.min_g1 = g1
                self.min_g1_theta = new_theta
        if valid_flag != 0 :
            gradient = 1.0/valid_flag * gradient
        else:
            gradient = None
        return gradient, gradient_flag
    
    def fine_grained_binary_search_local(self, theta, init_lbd = 1.0, tol=1e-5):
        
        lbd = init_lbd
        ori_confi, ori_label = self.classify(image_to_vector(self.model_name, self.dataset_name, self.image_ori+lbd*self.best_MASK*theta*255), 'opt')
        if ori_label != self.targeted:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            while self.classify(image_to_vector(self.model_name, self.dataset_name, self.image_ori+lbd_hi*self.best_MASK*theta*255), 'opt')[1] != self.targeted:
                lbd_hi = lbd_hi*1.01
                if lbd_hi > 400:
                    return float('inf')
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            while self.classify(image_to_vector(self.model_name, self.dataset_name, self.image_ori+lbd_lo*self.best_MASK*theta*255), 'opt')[1] == self.targeted :
                lbd_lo = lbd_lo*0.99
        
        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            if self.classify(image_to_vector(self.model_name, self.dataset_name, self.image_ori+lbd_mid*self.best_MASK*theta*255), 'opt')[1] == self.targeted:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi
    
    
    def update_direction_ORI(self, gradient, iteration_indice, step_num=15):        
        min_theta = self.theta
        min_g2 = self.g2
        
        # enlarge alpha
        for _ in range(step_num):
            increase_start = time.time()
            new_theta = (self.theta - self.alpha * gradient) * self.MASK
            new_theta = new_theta / torch.norm(new_theta)
            new_g2 = self.fine_grained_binary_search_local(new_theta, min_g2, tol=self.beta/500)
            
            self.alpha = self.alpha * 2
            self.attack_update_logger.log({
                'process': 'update_increase',
                'iteration_idx': '%4d'%iteration_indice, 
                'idx': '{}-{}'.format(_, 15),
                'alpha': '%.4f'%self.alpha,
                'ori_g2': '%.4f'%min_g2,
                'new_g2': '%.4f'%new_g2,
                'beta': '%.4f'%self.beta,
                'counts': '%4d'%self.opt_counts,
                'search_time(mins)':(time.time()-increase_start)/60.0
                })
            if new_g2 < min_g2:
                min_theta = new_theta 
                min_g2 = new_g2
            else:
                break
        
        # smaller alpha
        if min_g2 >= self.g2:
            for _ in range(step_num):
                self.alpha = self.alpha * 0.25
                new_theta = (self.theta - self.alpha * gradient) * self.MASK
                new_theta = new_theta / torch.norm(new_theta)
                new_g2 = self.fine_grained_binary_search_local(new_theta, min_g2, tol=self.beta/500)
            
                self.attack_update_logger.log({
                    'process': 'update_increase',
                    'iteration_idx': '%4d'%iteration_indice, 
                    'idx': '{}-{}'.format(_, 15),
                    'alpha': '%.4f'%self.alpha,
                    'ori_g2': '%.4f'%min_g2,
                    'new_g2': '%.4f'%new_g2,
                    'beta': '%.4f'%self.beta,
                    'counts': '%4d'%self.opt_counts,
                    'search_time(mins)':(time.time()-increase_start)/60.0
                    })
                if new_g2 < min_g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                    break
                    
        if min_g2 <= self.min_g1:
            self.theta, self.g2 = min_theta, min_g2
        else:
            self.theta, self.g2 = self.min_g1_theta, self.min_g1
        
        if self.g2 < self.g_theta:
            self.best_theta = self.theta
            self.g_theta = self.g2
            
        #self.theta, self.g2 = self.best_theta,self.g_theta

    def attack(self):
        # define logger
        self.attack_gradient_logger = Logger(
                os.path.join(self.output_path, 'attack_gradient_{}.log'.format(self.attack_idx)),
                ['process', 'iteration_idx', 'idx', 'g1', 'g2', 'beta', 'counts', 'search_time(mins)'])
        self.attack_update_logger = Logger(
                os.path.join(self.output_path, 'attack_update_{}.log'.format(self.attack_idx)),
                ['process', 'iteration_idx', 'idx', 'alpha', 'ori_g2', 'new_g2', 'beta', 'counts', 'search_time(mins)'])
        if self.salient_region:
            self.initialize_salient_region_mask()
        if self.del_frame:
            if self.bound:
                print ('initialize_from train dataset!')
                self.initialize_from_train_dataset_del_frame_bound()
            else:
                print ('initialize_from train dataset!')
                self.initialize_from_train_dataset_del_frame()
        else:
            print ('initialize_from train dataset!')
            self.initialize_from_train_dataset_baseline()
        print ('Do update')
        # update
        if self.g_theta != float('inf'):
            initialize_confi, initialize_indice = self.classify(image_to_vector(self.model_name, self.dataset_name, self.image_ori+ self.best_MASK * self.best_theta * self.g_theta  * 255))
            print ('IDX-{}, After initialize form train dataset, the adv-gt-adv is {}-{}-{}.'.format(self.attack_idx, initialize_indice, self.y0, self.targeted))
            for iterat in range(self.iterations):
                gradient, gradient_flag = self.estimate_gradient(iterat)
                if gradient_flag!= self.q:
                    self.update_direction_ORI(gradient, iterat, step_num=15)
                    ori_confi, ori_label = self.classify(image_to_vector(self.model_name, self.dataset_name, self.image_ori+ self.best_MASK * self.best_theta*self.g_theta*255))
                    print ('IDX-{}, Iterations{}-{}, adv-gt-adv {}-{}-{}'.format(self.attack_idx, iterat+1, self.iterations, ori_label, self.y0, self.targeted))
                    #adv_image = vector_to_image(self.model_name, image_to_vector(self.model_name, self.image_ori + self.best_theta*self.g_theta*255))
                    
                    #if torch.mean(torch.abs(adv_image - self.image_ori)) <= 12.75:
                    #    break

                    if self.g_theta < 20:
                         break
                        
                    if iterat >300 and self.g_theta < 45:
                        break
                    if self.alpha < 1e-4/16:
                        self.alpha = 2.0
                        self.beta = self.beta * 0.1
                        if self.beta < 0.0005/16:
                            break         
                    
                else:
                    self.beta = self.beta * 0.1
                    if self.beta < 0.0005/16:
                        break

                
            adv_confi, adv_indice = self.classify(image_to_vector(self.model_name, self.dataset_name, self.image_ori+ self.best_MASK * self.g_theta * self.best_theta * 255), 'opt')
            if adv_indice == self.targeted:
                self.success = True
                self.adv_confi = adv_confi
                self.adv_indice = adv_indice
                self.adv_image = outer_deal_image(self.image_ori + self.best_MASK * self.best_theta * self.g_theta * 255).type(torch.IntTensor)
                self.adv_image = vector_to_image(self.model_name, self.dataset_name, image_to_vector(self.model_name, self.dataset_name, self.image_ori + self.best_MASK * self.best_theta*self.g_theta*255)).type(torch.FloatTensor).cuda()
                self.P = torch.mean(torch.abs(self.adv_image - self.image_ori))
            else:
                self.success = False
        else:
            print ('Initialize False!')
            self.success = False