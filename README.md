# Heuristic-Black-box-Adversarial-Attack-On-Videos
Paper code: “Heuristic Black-box Adversarial Attacks on Video Recognition Models”[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6918/6772).

# Dataset
UCF-101 and HMDB-51 datasets are preprocessing by the methods in [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch).  
"dataset_load.py" file loads datasets for the specified model.
## Dataset-C3D
Parameters "root_path", "video_path", "annotation_path" need to be customized in "datasets/c3d_dataset/c3d_opt.py".  
* Generate the parameters file in pickle format
```bash
python c3d_opt.py
```  

# Model
C3D and LRCN models are from [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch) and [video_adv](https://github.com/yanhui002/video_adv/tree/master/models/inception) respectively.  
"models_load.py" file loads models for the specified model and dataset.

## C3D
### C3D-UCF101
Parameter "pretrain_path" is the path of the pretrain model in "video_cls_models/c3d/ucf101_opts.py".    
Download [here](https://drive.google.com/open?id=1DmI6QBrh7xhme0jOL-3nEutJzesHZTqp).
* Generate the parameters file in pickle format
```bash
python ucf101_opts.py
```
* Use the path of the parameters file to specify the line 19 in 'video_cls_models/c3d/c3d.py'.
### C3D-HMDB51
Parameter "pretrain_path" is the path of the pretrain model in "video_cls_models/c3d/hmdb51_opts.py".  
Download [here](https://drive.google.com/open?id=1GWP0bAff6H6cE85J6Dz52in6JGv7QZ_u).
* Generate the parameters file in pickle format
```bash
python hmdb51_opts.py
```
* Use the path of the parameters file to specify the line 15 in 'video_cls_models/c3d/c3d.py'.

# Attack Method
```bash
python main.py --dataset_name <hmdb51/ucf101> --model_name <lrcn/c3d> --target <True/False> --del_frame <True/False> --bound <True/False> --bound_threshold <int> --salient_region <True/False> --spatial_ratio <0~1> --spe_id <int>
```
* target: true/false means targeted/untargeted attack.
* del_frame: if or not use the temporal sparse.
* bound: if or not use the perturbation bound ω.
* bound_threshold: the perturbation bound ω.
* salient_region: if or not use the spatial sparse.
* spatial_ratio: the area ratio of salient region φ.
* spe_id: the id of the sample in the dataset.
