import os
import time
import glob
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from autoaugment import ImageNetPolicy

class UCF101(Dataset):
    def __init__(self, model, frames_path, labels_path, frame_size, sequence_length, setname='train',
        random_pad_sample=True, pad_option='default', 
        uniform_frame_sample=True, random_start_position=True, max_interval=7, random_interval=True):

        self.sequence_length = sequence_length

        # pad option => using for _add_pads function
        self.random_pad_sample = random_pad_sample
        assert pad_option in ['default', 'autoaugment'], "The pad option '{}' is not valid, you can try 'default' or 'autoaugment' pad option"
        self.pad_option = pad_option

        # frame sampler option => using for _frame_sampler function
        self.uniform_frame_sample = uniform_frame_sample
        self.random_start_position = random_start_position
        self.max_interval = max_interval
        self.random_interval = random_interval

        # read a csv file that already separated by splitter.py
        assert setname in ['train', 'test'], "'{}' setname is invalid".format(setname)
        if setname == 'train':
            csv = open(os.path.join(labels_path, 'train.csv'))
        if setname == 'test':
            csv = open(os.path.join(labels_path, 'test.csv'))
        self.data_paths = []
        
        # this value will using for CategoriesSampler class
        self.classes = [] # ex. [1, 1, 1, ..., 61, 61, 61]
        
        self.labels = {} # ex. {HulaHoop: 1, JumpingJack: 2, ..., Hammering: 61}
        lines = csv.readlines()
        for line in lines:
            label, folder_name = line.rstrip().split(',')
            action = folder_name.split('_')[1]
            self.data_paths.append(os.path.join(frames_path, folder_name))
            self.classes.append(int(label))
            self.labels[action] = int(label)
        csv.close()

        self.num_classes = len(self.labels)

        # select normalize value
        if model == "resnet":
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if model == "r2plus1d":
            normalize = transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])

        # transformer in training phase
        if setname == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((frame_size, frame_size)),
                transforms.ToTensor(),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4
                ),
                Lighting(alphastd=0.1, eigval=[0.2175, 0.0188, 0.0045],
                                        eigvec=[[-0.5675, 0.7192, 0.4009],
                                                [-0.5808, -0.0045, -0.8140],
                                                [-0.5836, -0.6948, 0.4203]]
                ),
                normalize,
            ])
        else:
        # transformer in validation or testing phase
            self.transform = transforms.Compose([
                transforms.Resize((frame_size, frame_size)),
                transforms.ToTensor(),
                normalize,
            ])
        
        # autoaugment transformer for insufficient frames in training phase
        self.transform_autoaugment = transforms.Compose([
            transforms.Resize((frame_size, frame_size)),
            ImageNetPolicy(),
            transforms.ToTensor(),
            normalize,
        ])

    def __len__(self):
        return len(self.data_paths)

    def _add_pads(self, sorted_frames_path):
        # get sorted frames length to list
        sequence = np.arange(len(sorted_frames_path))
        
        if self.random_pad_sample:
            # random sampling of pad
            add_sequence = np.random.choice(sequence, self.sequence_length - len(sequence))
        else:
            # repeated of first pad
            add_sequence = np.repeat(sequence[0], self.sequence_length - len(sequence))
        
        # sorting the pads
        sequence = sorted(np.append(sequence, add_sequence, axis=0))

        # transform to Tensor
        if self.pad_option == 'autoaugment':
            datas = [self.transform_autoaugment(Image.open(sorted_frames_path[s])) for s in sequence]
        else:
            datas = [self.transform(Image.open(sorted_frames_path[s])) for s in sequence]

        return datas

    def _frame_sampler(self, sorted_frames_path):
        # get sorted frames length to list
        sorted_frames_length = len(sorted_frames_path)

        # set interval for uniform sampling of frames
        interval = sorted_frames_length // self.sequence_length
        interval = 1 if interval == 0 else interval
        interval = self.max_interval if interval >= self.max_interval else interval
        if self.random_interval:
            # set interval randomly
            interval = np.random.permutation(np.arange(start=1, stop=interval + 1))[0]

        # set start position for uniform sampling of frames
        if self.random_start_position:
            start_position = np.random.randint(0, sorted_frames_length - (interval * self.sequence_length) + 1)
        else:
            start_position = 0

        # sampling frames
        if self.uniform_frame_sample:
            sequence = range(start_position, sorted_frames_length, interval)[:self.sequence_length]
        else:
            sequence = sorted(np.random.permutation(np.arange(sorted_frames_length))[:self.sequence_length])
        datas = [self.transform(Image.open(sorted_frames_path[s])) for s in sequence]
        
        return datas
    
    def __getitem__(self, index):
        # get frames and sort
        data_path = self.data_paths[index % len(self)]
        sorted_frames_path = sorted(glob.glob(data_path+"/*.jpg"), key=lambda path: int(path.split(".jpg")[0].split("\\" if os.name == 'nt' else "/")[-1]))
        sorted_frames_length = len(sorted_frames_path)
        assert sorted_frames_length != 0, "'{}' Path is not exist or no frames in there.".format(data_path)

        # we may be encounter that error such as
        # 1. when insufficient frames of video rather than setted sequence length, _add_pads function will be solve this problem
        # 2. when video has too many frames rather than setted sequence length, _frame_sampler function will be solve this problem
        if sorted_frames_length < self.sequence_length:
            datas = self._add_pads(sorted_frames_path)
        else:
            datas = self._frame_sampler(sorted_frames_path)
        
        datas = torch.stack(datas)
        labels = self.labels[data_path.split("_")[-3]]
        return datas, labels

class CategoriesSampler():
    def __init__(self, labels, iter_size, way, shot, query):
        self.iter_size = iter_size
        self.way = way
        self.shots = shot + query

        labels = np.array(labels)
        self.indices = []
        for i in range(1, max(labels) + 1):
            index = np.argwhere(labels == i).reshape(-1)
            index = torch.from_numpy(index)
            self.indices.append(index)

    def __len__(self):
        return self.iter_size
    
    def __iter__(self):
        for i in range(self.iter_size):
            batchs = []
            classes = torch.randperm(len(self.indices))[:self.way] # bootstrap(class)
            for c in classes:
                l = self.indices[c]
                pos = torch.randperm(len(l))[:self.shots] # bootstrap(shots)
                batchs.append(l[pos])
            batchs = torch.stack(batchs).t().reshape(-1)
            yield batchs

# original code: https://github.com/eladhoffer/convNet.pytorch/blob/master/preprocess.py
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone().mul(alpha.view(1, 3).expand(3, 3)).mul(self.eigval.view(1, 3).expand(3, 3)).sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))