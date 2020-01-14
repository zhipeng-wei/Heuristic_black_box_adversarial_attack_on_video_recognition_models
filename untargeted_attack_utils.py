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
# import torch_dct as dct

from base_utils import *
from salient_region_utils import *

class Attack_base(object):
    def __init__(self, model, train_dataset, attack_idx, x0, y0, output_path, model_name, init_samples, num_samples=100, alpha=0.2, beta=0.01, iterations=1000, d=15, q=15, del_frame=False, bound=False, bound_threshold=8, salient_region=False, spatial_mode=0, spatial_ratio=0.6):
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
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.num_samples = num_samples
        self.d = d
        self.q = q
        self.output_path = output_path
        self.model_name = model_name
        
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
        self.image_ori = vector_to_image(self.model_name, self.x0)
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
        self.MASK = MASKs
    
    def frames_to_mask(self, frame_indices):
        mask = torch.zeros(self.x0.size())
        if self.model_name == 'c3d':
            mask[:,frame_indices,:,:] = 1
        else:
            mask[frame_indices,:,:,:] = 1
        return mask.cuda() * self.MASK
    
    def get_bounding_value(self, i, frame_indices, image_adv, g_theta):
        tmp_mask = self.frames_to_mask(frame_indices)
        DIV = 255
        tmp_vector = image_to_vector(self.model_name, (image_adv-self.image_ori)*tmp_mask+self.image_ori)
        this_prob, this_pre = self.classify(tmp_vector, 'query')
        if this_pre != self.y0:
            theta = (image_adv-self.image_ori)/DIV * tmp_mask
            initial_lbd = torch.norm(theta)
            theta = theta/initial_lbd
            binary_search_start = time.time()
            lbd = self.fine_grained_binary_search(theta, initial_lbd, g_theta)
            if lbd == float('inf'):
                return None
            tmp_noise = theta * lbd * DIV
            all_nums = int(torch.sum(tmp_mask.reshape(-1)).item())
            valid_indices = torch.argsort(-tmp_mask.reshape(-1))[:all_nums]
            tmp_p = torch.mean(torch.abs(tmp_noise.reshape(-1)[valid_indices]))
            return (frame_indices, initial_lbd, lbd, theta, binary_search_start, tmp_p, this_pre)
        else:
            return None
        
    def initialize_from_train_dataset_del_frame_bound(self):
        '''
        Initialize theta(direction) and g2(distance along with the direction)
        '''
        attack_initialize_logger = Logger(
                os.path.join(self.output_path, 'attack_initialize_from_train_{}.log'.format(self.attack_idx)),
                ['process', 'attack_idx', 'initial_idx', 'ori_lbd', 'cur_lbd', 'counts', 'this_label', 'ori_label', 'P', 'masked_frames', 'search_time(mins)'])
        
        outer_best_p = float('inf')
        best_theta, g_theta = None, float('inf')
        for i in self.samples:
            xi, yi = self.train_dataset[i]
            xi = xi.cuda()
            vector_noise = (xi-self.x0) * self.MASK
            vector_adv = vector_noise + self.x0
            this_prob, this_pre = self.classify(vector_adv, 'query')
            if this_pre != self.y0:
                
                image_adv = vector_to_image(self.model_name, vector_adv)
                del_frame_sequences = loop_del_frame_sort_sequence(image_adv.cpu(), self.image_ori.cpu(), self.y0, self.model, self.model_name, self.MASK)
                if not del_frame_sequences:
                    continue
                begin_frames = [i for i in range(self.seq_len)]
                re = self.get_bounding_value(i, begin_frames, image_adv, g_theta)
                
                if not re:
                    continue
                frame_indices, initial_lbd, lbd, theta, binary_search_start, tmp_p, this_pre = re
                
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
                    tmp_frames = [i for i in inner_frames if i != del_frame]                    
                    re = self.get_bounding_value(i, tmp_frames, image_adv, g_theta)
                    if  re:
                        frame_indices, initial_lbd, lbd, theta, binary_search_start, tmp_p, this_pre = re
                        if inner_p >= self.bound_threshold:
                            if tmp_p < inner_p:
                                inner_frames = tmp_frames
                                inner_p = tmp_p
                                inner_lbd = lbd
                                inner_theta = theta
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
                        'this_label': '%4d'%self.y0,
                        'ori_label': '%4d'%self.y0,
                        'P' : '%4d'%inner_p,
                        'masked_frames': '-'.join([str(i) for i in inner_frames]),
                        'search_time(mins)': (time.time()-binary_search_start)/60.0
                        })
                if inner_p < outer_best_p:
                    best_theta, g_theta = inner_theta, inner_lbd
                    outer_best_p = inner_p
                    self.best_frame_indices = inner_frames 
        
        self.best_MASK = self.frames_to_mask(self.best_frame_indices)
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
            vector_noise = (xi-self.x0) * self.MASK
            vector_adv = vector_noise + self.x0
            this_prob, this_pre = self.classify(vector_adv, 'query')
            if this_pre != self.y0:
                image_adv = vector_to_image(self.model_name, vector_adv)
                del_flag, MASK, frame_indices, this_query  = loop_del_frame_sort(image_adv.cpu(), self.image_ori.cpu(), self.y0, self.model, self.model_name, self.MASK)
                if del_flag:
                    MASK = MASK * self.MASK
                else:
                    MASK = self.MASK
                self.query_counts = self.query_counts + this_query
                DIV = 255
                theta = (image_adv-self.image_ori)/DIV * MASK
                initial_lbd = torch.norm(theta)
                theta = theta/initial_lbd
                binary_search_start = time.time()
                
                #vector_adv_tmp = image_to_vector(self.model_name, theta*initial_lbd*DIV+self.image_ori)
                #this_prob, this_pre = self.classify(vector_adv_tmp, 'query')
                #if this_pre == self.y0:
                    #break
                
                lbd = self.fine_grained_binary_search(theta, initial_lbd, g_theta)
                attack_initialize_logger.log({
                    'process': 'initialize_theta',
                    'attack_idx': '%4d'%self.attack_idx,
                    'initial_idx': '%4d'%i,
                    'ori_lbd': '%.4f'%initial_lbd,
                    'cur_lbd': '%.4f'%lbd,
                    'counts': '%4d'%self.query_counts,
                    'this_label': '%4d'%this_pre,
                    'ori_label': '%4d'%self.y0,
                    'masked_frames': '-'.join([str(i) for i in frame_indices]),
                    'search_time(mins)': (time.time()-binary_search_start)/60.0
                    })
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    self.best_MASK = MASK
                    self.best_frame_indices = frame_indices
                    
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
            if this_pre != self.y0:
                image_adv = vector_to_image(self.model_name, xi)
                DIV = 255
                theta = (image_adv-self.image_ori)/DIV * self.MASK
                initial_lbd = torch.norm(theta)
                theta = theta/initial_lbd
                binary_search_start = time.time()
                lbd = self.fine_grained_binary_search(theta, initial_lbd, g_theta)
                attack_initialize_logger.log({
                    'process': 'initialize_theta',
                    'attack_idx': '%4d'%self.attack_idx,
                    'initial_idx': '%4d'%i,
                    'ori_lbd': '%.4f'%initial_lbd,
                    'cur_lbd': '%.4f'%lbd,
                    'counts': '%4d'%self.query_counts,
                    'this_label': '%4d'%this_pre,
                    'ori_label': '%4d'%self.y0,
                    'masked_frames': 'all',
                    'search_time(mins)': (time.time()-binary_search_start)/60.0
                    })
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
        self.best_theta, self.g_theta = best_theta, g_theta
        self.theta, self.g2 = best_theta, g_theta
 
    
    
    def fine_grained_binary_search(self, theta, initial_lbd, current_best):
        if initial_lbd > current_best: 
            ori_confi, ori_label = self.classify(image_to_vector(self.model_name, self.image_ori+current_best*theta*255), 'query')
            if ori_label == self.y0:
                return float('inf')
            lbd = current_best
        else:
            lbd = initial_lbd
            
        lbd_hi = lbd
        lbd_lo = 0.0
  
        while (lbd_hi - lbd_lo) > 1e-2:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            ori_confi, ori_label = self.classify(image_to_vector(self.model_name, self.image_ori+lbd_mid*theta*255), 'query')
            if ori_label != self.y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
  
        return lbd_hi


    def estimate_gradient(self, iteration_indice):
        gradient = torch.zeros(self.x0.size()).cuda()
        self.min_g1_theta, self.min_g1 = None, float('inf')
        gradient_flag = 0
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
            gradient += self.d * (g1-self.g2)/self.beta * u
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
        gradient = 1.0/self.q * gradient
        return gradient, gradient_flag
    
    def fine_grained_binary_search_local(self, theta, init_lbd = 1.0, tol=1e-2):
        lbd = init_lbd
        ori_confi, ori_label = self.classify(image_to_vector(self.model_name, self.image_ori+lbd*theta*255), 'opt')
        if ori_label == self.y0:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            while self.classify(image_to_vector(self.model_name, self.image_ori+lbd_hi*theta*255), 'opt')[1] == self.y0:
                lbd_hi = lbd_hi*1.01
                if lbd_hi > 320:
                    return float('inf')
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            while self.classify(image_to_vector(self.model_name, self.image_ori+lbd_lo*theta*255), 'opt')[1] != self.y0 :
                lbd_lo = lbd_lo*0.99
        
        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            if self.classify(image_to_vector(self.model_name, self.image_ori+lbd_mid*theta*255), 'opt')[1] != self.y0:
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
            new_g2 = self.fine_grained_binary_search_local(new_theta, min_g2, tol=1e-2)
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
                new_g2 = self.fine_grained_binary_search_local(new_theta, min_g2, tol=1e-2)
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
                if new_g2 < self.g2:
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
    

    def attack(self):
        # define logger
        self.attack_gradient_logger = Logger(
                os.path.join(self.output_path, 'attack_gradient_{}.log'.format(self.attack_idx)),
                ['process', 'iteration_idx', 'idx', 'g1', 'g2', 'beta', 'counts', 'search_time(mins)'])
        self.attack_update_logger = Logger(
                os.path.join(self.output_path, 'attack_update_{}.log'.format(self.attack_idx)),
                ['process', 'iteration_idx', 'idx', 'alpha', 'ori_g2', 'new_g2', 'beta', 'counts', 'search_time(mins)'])
        if self.salient_region:
            print ('Runing with salient region.')
            self.initialize_salient_region_mask()
        if self.del_frame:
            if self.bound:
                print ('Runing with del_frame, bound')
                self.initialize_from_train_dataset_del_frame_bound()
            else:
                print ('Runing with del_frame')
                self.initialize_from_train_dataset_del_frame()
        else:
            print ('Runing with baseline')
            self.initialize_from_train_dataset_baseline()
        
        # update
        if self.g_theta != float('inf'):
            initialize_confi, initialize_indice = self.classify(image_to_vector(self.model_name, self.image_ori+ self.g_theta * self.best_theta * 255))
            print ('IDX-{}, After initialize form train dataset, the adv-gt is {}-{}.'.format(self.attack_idx, initialize_indice, self.y0))
            for iterat in range(self.iterations):
                gradient, gradient_flag = self.estimate_gradient(iterat)
                if gradient_flag!= self.q:
                    self.update_direction_ORI(gradient, iterat, step_num=15)
                    ori_confi, ori_label = self.classify(image_to_vector(self.model_name, self.image_ori+self.best_theta*self.g_theta*255))
                    print ('IDX-{}, Iterations{}-{}'.format(self.attack_idx, iterat+1, self.iterations), ori_label, self.y0)
                    if self.alpha < 1e-4:
                        self.alpha = 1.0
                        self.beta = self.beta * 0.1
                        if self.beta < 0.0001:
                            break
                else:
                    print ('estimate_gradient, all_error!')
                    break

            adv_confi, adv_indice = self.classify(image_to_vector(self.model_name, self.image_ori+ self.g_theta * self.best_theta * 255), 'opt')
            if adv_indice != self.y0:
                self.success = True
                self.adv_confi = adv_confi
                self.adv_indice = adv_indice
                self.adv_image = vector_to_image(self.model_name, image_to_vector(self.model_name, self.image_ori + self.best_theta*self.g_theta*255))
                self.P = torch.mean(torch.abs(self.adv_image - self.image_ori))
            else:
                self.success = False
        else:
            print ('Initialize False!')
            self.success = False