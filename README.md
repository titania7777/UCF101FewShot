# UCF101 Few-shot Action Recognition(metric-based)
sample code for few-shot action recognition on UCF101

```UCF101.py``` sampler supports autoaugment[1] when scarcity of frames in video(optional).

## Requirements
*   torch>=1.6.0
*   torchvision>=0.7.0
*   tensorboard>=2.3.0

## Usage
download and extract frame from UCF101 videos. [UCF101 Frame Extractor](https://github.com/titania7777/UCF101Frameextractor)

split dataset for few-shot learning(if you already has csv files then you can skip this step)
```
python splitter.py --frames-path /path/to/frames --labels-path /path/to/labels --save-path /path/to/save
```
train(resnet18)
```
python train.py --frames-path /path/to/frames --save-path /path/to/save --tensorboard-path /path/to/tensorboard --model resnet --uniform-frame-sample --frame-size 168 --way 5 --shot 1 --query 5
```
train(r2plus1d18)
```
python train.py --frames-path /path/to/frames --save-path /path/to/save --tensorboard-path /path/to/tensorboard --model r2plus1d --uniform-frame-sample --metric cosine --way 5 --shot 1 --query 5
```
test(resnet18)
```
python test.py --frames-path /path/to/frames --load-path /path/to/load --use-best --model resnet --frame-size 168 --way 5 --shot 1 --query 5
```
test(r2plus1d18)
```
python test.py --frames-path /path/to/frames --load-path /path/to/load --use-best --model r2plus1d --metric cosine --way 5 --shot 1 --query 5
```

## Settings and Results
```device information```  
GPU: RTX 2080 Ti(11GB)  

```data settings```  
train class: 71 (9473 videos)  
test(val) class: 30 (3847 videos)  

```Common option settings```  
frame size: 112(r2plus1d), 168(resnet)  
num epochs: 30 
train iter size: 100  
val iter size: 200  
metric: cosine  
random pad sample: False  
pad option: default  
uniform frame sample: True  
random start position: False  
max interval: 7  
random interval: False  
sequence length: 35  
learning rate: 1e-4(r2plus1d), 5e-4(resnet)  
scheduler step: 10  
scheduler gamma: 0.9  
way: 5  
shot: 1  
query: 5  

```require video memory```  
resnet: about 7538 MB  
r2plus1d: about 10042 MB  

option | Accuracy
-- | -- 
resnet18 | 69.15 ±1.20
r2plus1d18(without training)  | 92.82 ±0.74
r2plus1d18  | 94.29 ±0.67

## ```UCF101.py``` Options
### common options
1. model: choose for different normalization value of model
2. frames_path: frames path
3. labels_path: labels path
4. frame_size: frame size(width and height are should be same)
5. sequence_length: number of frames
6. setname: sampling mode, if this mode is 'train' then the sampler read a 'train.csv' file to load train dataset [default: 'train', others: 'test']
### pad options
7. random_pad_sample: sampling frames from existing frames with randomly when frames are insufficient, if this value is False then only use first frame repeatedly [default: True, other: False]
8. pad_option: when adds some pad for insufficient frames of video, if this value is 'autoaugment' then pads will augmented by autoaugment policies [default: 'default', other: 'autoaugment']
### frame sampler options
9. uniform_frame_sample: sampling frames with same interval, if this value is False then sampling frames with ignored interval [default: True, other: False]
10. random_start_position: decides the starting point with randomly by considering the interval, if this value is False then starting point is always 0 [default: True, other, False]
11. max_interval: setting of maximum frame interval, if this value is high then probability of missing sequence of video is high [default: 7]
12. random_interval: decides the interval value with randomly, if this value is False then use a maximum interval [default: True, other: False]

## CategoriesSampler Options in ```UCF101.py```
1. labels: this parameter receive of classes in csv files, so this value must be ```UCF101.classes```
2. iter_size: number of iteration per episodes(total episode = epochs * iter_size)
3. way: number of way(number of class)
4. shot: number of shot
5. query: number of query  
*way, shot, query => we follow episodic training stratiegy[2]

## references
-------------
[1] Ekin D. Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, Quoc V. Le, "AutoAugment: Learning Augmentation Strategies From Data", Computer Vision and Pattern Recognition(CVPR), 2019, pp. 113-123  
[2] Vinyals, Oriol and Blundell, Charles and Lillicrap, Timothy and kavukcuoglu, koray and Wierstra, Daan, "Matching Networks for One Shot Learning", Neural Information Processing Systems(NIPS), 2016, pp. 3630-3638
