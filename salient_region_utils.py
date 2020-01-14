import cv2
import numpy as np
import random
import torch

#************************************************
#cv2 function
#************************************************
def get_cv2_func(mode):
    if mode == 0:
        print ('Run with StaticSaliencySpectralResidual!')
        return cv2.saliency.StaticSaliencySpectralResidual_create()
    elif mode == 1:
        print ('Run with StaticSaliencyFineGrained!')
        return cv2.saliency.StaticSaliencyFineGrained_create()
    elif mode == 2:
        return None
    
def SpectralResidual(cv2_func, image, ratio, model_name):
    if model_name == 'c3d':
        image = np.array(image.permute(1,2,0).cpu()).astype(np.uint8)        
    elif model_name == 'lrcn':
        image = np.array(image.cpu()).astype(np.uint8)
    elif model_name == 'flownet':
        image = np.array(image.permute(1,2,0).cpu()).astype(np.uint8)
    if cv2_func:
        (success, saliencyMap) = cv2_func.computeSaliency(image)
        flat_saliency = saliencyMap.flatten()
        MASK = np.zeros_like(saliencyMap)
        flat_MASK = MASK.flatten()
        indices = np.argsort(-flat_saliency)
        
        useful_indices = indices[: int(len(indices)*ratio)]
        flat_MASK[useful_indices] = 1
        MASK = np.reshape(flat_MASK, saliencyMap.shape)
        MASK = np.stack([MASK, MASK, MASK], axis=2)
        MASK = torch.from_numpy(MASK)
    else:
        print ('Run with Random')
        MASK = RandomSpatial(image, ratio, model_name)
    return MASK


def RandomSpatial(image, ratio, model_name):
    MASK = np.zeros_like(image)
    flat_MASK = MASK.flatten()
    random.seed(1024)
    indices = random.sample([i for i in range(len(flat_MASK))], int(len(flat_MASK)*ratio))
    flat_MASK[indices] = 1
    MASK = np.reshape(flat_MASK, image.shape)
    MASK = torch.from_numpy(MASK)
    return MASK

#************************************************
#other functions
#************************************************