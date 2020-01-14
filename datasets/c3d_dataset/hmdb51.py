import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import numpy as np 
from utils import load_value_file, calOpticalFlow


def pil_loader(path):
    # 使用pil加载图片，返回RGB排列的矩阵
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    # 使用accimage来加载图片
    # 另外一种图片打开方式
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    # torchvision.get_image_backend()，得到用来加载图片的包的名字
    # accimage比PIL的速度要快，但是在一些操作中不支持。
    # gets the name of the package used to load images
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    # 通过读取指定frame_indices上的图片组成的列表，作为视频表示
    # 在经过预处理后，每一个avi视频都使用avi名字的文件夹表示，文件夹中表示各个frame
    # 且使用image_x.jpg来表示
    # video_loader是给定video_dir_path，以及所需要的frame_indices，通过image_loader
    # 来对该视频中指定的帧图片进行加载，组成四维的列表N_frames * height * width * N_channels
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video
    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)
    # 使得video_loader函数拥有默认参数值image_loader=image_loader.


def load_annotation_data(data_file_path):
    # 需要看一下hmdb51_json.py
    # 读取经由hmdb51_json.py文件生成的json文件
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    # 根据data中的label来由0开始生成数值
    # 将label转化为数值
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    # 同样根据json文件来保存内容
    # 生成label/file_name的video列表，以及label的annotation列表
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))
            annotations.append(value['annotations'])
            
    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):
    # 生成数据列表，列表中的元素为{'video':video_path, 'segment':[begin_t, end_t],\
    # 'n_frames':n_frames, 'video_id':video_names, 'label':, 'frame_indices'}
    # 其中frmae_indices表示所选取的frame的索引值。
    # sample_duration： 采样的间隔
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            continue

        n_frames_file_path = os.path.join(video_path, 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i].split('/')[1]
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            # 表示的意思应该是选取视频中所有帧的样本
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            # because of optical flow, we modify the beginning is 2.
            for j in range(2, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, idx_to_class


class HMDB51(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 input_style = 'rgb',
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader,
                 flow_channels = 3):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)
        # 输入PIL图像，之后返回经过转化的版本，例如进行随机的切割旋转等，捕获空间特征
        self.spatial_transform = spatial_transform
        # 输入一系列帧的索引，返回经过转化的版本，例如optical flow。
        self.temporal_transform = temporal_transform
        # 对目标进行转化
        self.target_transform = target_transform
        # 给定路径以及帧的索引来读取的函数
        self.loader = get_loader()
        self.input_style = input_style
        self.flow_channels = flow_channels
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']
        
        frame_indices = self.data[index]['frame_indices']
        # 为了不让i为0，我们在对视频的帧进行随机截取的时候删去第一帧
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        
        # 由于生成的optical flow是隐式的，故我们不在这里生成opticalflow
        # 通过生成rgb的数据，在传入模型之前，再加上生成optical flow的步骤
        if self.input_style == 'flow':
            clip_flow = self.loader(path, [i-1 for i in frame_indices])
            flows = []
            for pre_image, next_image in zip(clip_flow, clip):
                flow = calOpticalFlow(np.array(pre_image), np.array(next_image), self.flow_channels)
                flows.append(Image.fromarray(np.uint8(flow)))
            clip = flows
        
        # 注意，此时clip为列表，其中保存的是np.array格式，而在torch.stack中需要将np.array转为tensor
        # spatial_transform
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)
