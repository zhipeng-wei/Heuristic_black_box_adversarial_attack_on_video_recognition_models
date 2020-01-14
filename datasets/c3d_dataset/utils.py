import numpy as np
# import cv2
import csv
from PIL import Image
import torch.nn as nn
import torch

# 生成原图 —— 生成optical flow —— 经过transforms ——— 作为模型输入


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value

def get_mean(norm_value=255, dataset='hmdb51'):
    assert dataset in ['hmdb51', 'ucf101']
    if dataset == 'hmdb51':
        return [
            95.4070 / norm_value, 93.4680 / norm_value, 82.1443 / norm_value
        ]
    elif dataset == 'ucf101':
        # Kinetics (10 videos for each class)
        return [
            101.2198/ norm_value, 97.5751 / norm_value,
            89.5303 / norm_value
        ]

def get_std(norm_value=255, dataset='hmdb51'):
    assert dataset in ['hmdb51', 'ucf101']
    if dataset == 'hmdb51':
        return [
            51.674248 / norm_value, 50.311924 / norm_value,
            49.48427 / norm_value
        ]
    elif dataset == 'ucf101':
        return [
            62.08429 / norm_value, 60.398968 / norm_value,
            59.187363 / norm_value
        ]
    
def generate_flows(inputs, channels):
    '''
    Convert RGB inputs to Optical Flow inputs.
    RGB inputs: batch_size * num_channels * num_frames * height * width.
    Optical Flow outputs: batch_size * num_channels * (num_frames-1) * height * width
    Parameters:
        inputs: the RGB inputs, type tenosr, shape is batch_size * num_channels * num_frames * height * width.
        channels: type int, the channels of Optical Flow outputs, for i3d is 2, for lrcn is 3.
    Return: 
        clips: type Image.Image
    '''
    # batch_size * num_channels * num_frames * height * width
    batch_size = inputs.size()[0]
    num_frames = inputs.size()[2]

    optical_flow = cv2.DualTVL1OpticalFlow_create()
    
    flows = torch.zeros(batch_size, num_frames-1, inputs.size()[3], inputs.size()[4], channels)
    for i in range(batch_size):
        this_video = inputs[i]
        this_video = this_video.permute([1,2,3,0])

        for frame in range(num_frames):
            if frame == 0:
                continue
            else:
                prev_img = np.uint8(np.array(this_video[frame-1]))
                next_img = np.uint8(np.array(this_video[frame]))
                flow_image = calOpticalFlow(prev_img, next_img, channels, optical_flow)
            flows[i][frame-1] = torch.from_numpy(np.uint8(flow_image))
    flows = flows.permute([0,4,1,2,3])
    return flows

def calOpticalFlow(frame1, frame2, channels, optical_flow):
    '''
    Reference: https://github.com/LisaAnne/lisa-caffe-public/blob/lstm_video_deploy/
               examples/LRCN_activity_recognition/create_flow_images_LRCN.m
    Params:
        frame1: frame_indice-1, height*width*3
        frame2: frame_indice, height*width*2
    Reture:
        flow: height*width*3, the third channel is created by calculating the flow magnitude.
              Resclaed to 0-255 for fit the pixel space with RGB.
    '''
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    nex = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = optical_flow.calc(prvs, nex, None)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    x,y,z = flow.shape
    if channels == 3:
        flow_image = np.zeros((x,y,3))
        flow_image[:,:,0:2] = flow
        flow_image[:,:,2] = mag
    elif channels == 2:
        flow_image = np.zeros((x,y,2))
        flow_image[:,:,0:2] = flow
    flow_image = cv2.normalize(flow_image,None,-128,128,cv2.NORM_MINMAX)
    flow_image = flow_image.astype(int)
    return flow_image
