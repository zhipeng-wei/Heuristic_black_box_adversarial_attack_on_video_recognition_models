import csv
import torch
import numpy as np

def outer_deal_image(image):
    image[image>255] = 255
    image[image<0] = 0
    return image

def outer_deal_vector(model_name, dataset_name, x):
    if model_name == 'c3d':
        if dataset_name == 'ucf101':
            means = [101.2198,  97.5751,  89.5303]
        elif dataset_name == 'hmdb51':
            means = [95.4070, 93.4680, 82.1443]
        # add (batch_size, num_channels, seq_len, height, width)
        for i in range(3):
            x[i,:,:,:] = x[i,:,:,:] + means[i]
        # outers 
        x[x>255] = 255
        x[x<0] = 0
        # reduce
        for i in range(3):
            x[i,:,:,:] = x[i,:,:,:] - means[i]
    elif model_name == 'lrcn':
        x = x * 255
        x[x>255] = 255
        x[x<0] = 0
        x = x/255
    elif model_name == 'flownet':
        x[x>255] = 255
        x[x<0] = 0
    return x

def vector_to_image(model_name, dataset_name, x):
    # convert vector's range to (0, 255)
    if model_name == 'c3d':
        if dataset_name == 'ucf101':
            means = [101.2198,  97.5751,  89.5303]
        elif dataset_name == 'hmdb51':
            means = [95.4070, 93.4680, 82.1443]
        # add (batch_size, num_channels, seq_len, height, width)
        for i in range(3):
            x[i,:,:,:] = x[i,:,:,:] + means[i]
        x[x>255] = 255
        x[x<0] = 0
    elif model_name == 'lrcn':
        x = x * 255
        x[x>255] = 255
        x[x<0] = 0
    elif model_name == 'flownet':
        x[x>255] = 255
        x[x<0] = 0
    return x

def image_to_vector(model_name, dataset_name, x):
    # convert (0-255) image to specify range
    if model_name == 'c3d':
        if dataset_name == 'ucf101':
            means = [101.2198,  97.5751,  89.5303]
        elif dataset_name == 'hmdb51':
            means = [95.4070, 93.4680, 82.1443]
        # add (batch_size, num_channels, seq_len, height, width)
        # outers 
        x[x>255] = 255
        x[x<0] = 0
        # reduce
        for i in range(3):
            x[i,:,:,:] = x[i,:,:,:] - means[i]
    elif model_name == 'lrcn':
        x[x>255] = 255
        x[x<0] = 0
        x = x/255
    elif model_name == 'flownet':
        x[x>255] = 255
        x[x<0] = 0
    return x

class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values, col + 'not in' + values.keys()
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

def frames_to_mask(train_image, all_frames, model_name):
    MASK = torch.zeros(train_image.size()).cuda()
    if model_name == 'c3d':
        MASK[:, all_frames, :, :] = 1
    else:
        MASK[all_frames, :, :, :] = 1
    return MASK

def mask_pre_query(train_image, test_image, model, mask_list, model_name, spatial_mask):
    MASK = torch.zeros(train_image.size())
    if model_name == 'c3d':
        MASK[:, mask_list, :, :] = 1    
    else:
        MASK[mask_list, :, :, :] = 1

    MASK = MASK * spatial_mask.cpu()
    adv_image = (train_image - test_image) * MASK + test_image
    adv_vec = image_to_vector(model_name, adv_image)
    values, indices = torch.sort(-torch.nn.functional.softmax(model(adv_vec.cuda().unsqueeze(0))), dim=1)
    confidence_prob, pre_label = -float(values[:,0]), int(indices[:,0])
    return confidence_prob, pre_label

def loop_del_frame_sort(train_image, test_image, test_label, model, model_name, spatial_mask, mode = 'untarget'):
    '''
    1.对每个帧进行一次剔除
    2.对此时帧列表进行预测，query = n
        a.如果满足对抗性，那么保存其得分
        b.如果不满足对抗性，证明该帧不可删除，保留该帧
    3.通过得分得到排序的帧索引值
    4.从最高分的开始删
        a.保证删除满足对抗性
        b.如果删除不满足对抗性就跳过到下一个
    5.直到得到最终的帧索引列表
    '''
    all_frames = [i for i in range(16)]
    query = 0
    score_dict = {}
    for i in all_frames:
        tmp_frames = [_ for _ in all_frames if _!=i]
        this_confi, this_label = mask_pre_query(train_image, test_image, model, tmp_frames, model_name, spatial_mask)
        query += 1
        if mode == 'untarget':
            if this_label == test_label:
                pass
            else:
                score_dict[i] = this_confi
        elif mode == 'target':
            if this_label != test_label:
                pass
            else:
                score_dict[i] = this_confi
    if score_dict:
        sorted_items = sorted(score_dict.items(), key=lambda x:-x[1])
    else:
        return False, frames_to_mask(train_image, [i for i in range(16)], model_name), all_frames, query
    for item in sorted_items:
        frame_indice, confi = item
        tmp_frames = [i for i in all_frames if i!=frame_indice]
        this_confi, this_label = mask_pre_query(train_image, test_image, model, tmp_frames, model_name, spatial_mask)
        if mode == 'untarget':
            if this_label != test_label:
                all_frames = tmp_frames
                continue
            else:
                pass
        elif mode == 'target':
            if this_label == test_label:
                all_frames = tmp_frames
                continue
            else:
                pass            
    return True, frames_to_mask(train_image, all_frames, model_name), all_frames, query


def loop_del_frame_sort_sequence(train_image, test_image, test_label, model, model_name, spatial_mask, mode = 'untarget'):
    '''返回经过排序的帧的列表'''
    all_frames = [i for i in range(16)]
    query = 0
    score_dict = {}
    for i in all_frames:
        tmp_frames = [_ for _ in all_frames if _!=i]
        this_confi, this_label = mask_pre_query(train_image, test_image, model, tmp_frames, model_name, spatial_mask)
        query += 1
        if mode == 'untarget':
            if this_label == test_label:
                pass
            else:
                score_dict[i] = this_confi
        elif mode == 'target':
            if this_label != test_label:
                pass
            else:
                score_dict[i] = this_confi
    if score_dict:
        sorted_items = sorted(score_dict.items(), key=lambda x:-x[1])
    else:
        return None
    sorted_frames = []
    for item in sorted_items:
        sorted_frames.append(item[0])
    return sorted_frames

            