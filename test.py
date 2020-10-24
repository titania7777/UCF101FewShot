import torch
import os
import sys
import argparse
import torch.nn.functional as F
from models import R2Plus1D, Resnet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from UCF101 import UCF101, CategoriesSampler
from utils import printer, mean_confidence_interval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-path", type=str, default="../Data/UCF101_frames/")
    parser.add_argument("--labels-path", type=str, default="./UCF101_few_shot_labels/")
    parser.add_argument("--load-path", type=str, default="./save/train1")
    parser.add_argument("--use-best", action="store_true")
    parser.add_argument("--frame-size", type=int, default=112)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--sequence-length", type=int, default=35)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--model", type=str, default='resnet')
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--query", type=int, default=5)
    args = parser.parse_args()

    # model check
    assert args.model in ['resnet', 'r2plus1d'], "'{}' model is invalid".format(setname)
    
    if args.use_best:
        load_path = os.path.join(args.load_path, "best.pth")
    else:
        load_path = os.path.join(args.load_path, "last.pth")
    
    # load_path check
    assert os.path.exists(load_path), "'{}' file is not exists !!".format(load_path)


    test_dataset = UCF101(
        model=args.model,
        frames_path=args.frames_path,
        labels_path=args.labels_path,
        frame_size=args.frame_size,
        sequence_length=args.sequence_length,
        setname='test',
        # pad option
        random_pad_sample=False,
        pad_option='default',
        # frame sampler option
        uniform_frame_sample=True,
        random_start_position=False,
        max_interval=7,
        random_interval=False,
    )
    print("[test] number of videos / classes: {} / {}".format(len(test_dataset), test_dataset.num_classes))
    test_sampler = CategoriesSampler(test_dataset.classes, 400, args.way, args.shot, args.query)
    
    # in windows has some issue when try to use DataLoader in pytorch, i don't know why..
    test_loader = DataLoader(dataset=test_dataset, batch_sampler=test_sampler, num_workers=0 if os.name == 'nt' else 4, pin_memory=True)
        
    if args.model == 'resnet':
        model = Resnet(
            way=args.way,
            shot=args.shot,
            query=args.query,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            bidirectional=args.bidirectional,
        )

    if args.model == 'r2plus1d':
        model = R2Plus1D(
            way=args.way,
            shot=args.shot,
            query=args.query,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(load_path))
    
    model.eval()
    total_loss = 0
    total_acc = 0
    print("test... {}-way {}-shot {}-query".format(args.way, args.shot, args.query))
    for e in range(1, args.num_epochs+1):
        test_acc = []
        test_loss = []
        for i, (datas, _) in enumerate(test_loader):
            datas = datas.to(device)
            pivot = args.way * args.shot
            
            shot, query = datas[:pivot], datas[pivot:]
            labels = torch.arange(args.way).repeat(args.query).to(device)
            # one_hot_labels = Variable(torch.zeros(args.way*args.query, args.way).scatter_(1, labels.view(-1, 1), 1)).to(device)

            pred = model(shot, query)

            # calculate loss
            loss = F.cross_entropy(pred, labels).item()
            test_loss.append(loss)
            total_loss = sum(test_loss)/len(test_loss)

            # calculate accuracy
            acc = 100 * (pred.argmax(1) == labels).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor).mean().item()
            test_acc.append(acc)
            total_acc = sum(test_acc)/len(test_acc)

            printer("test", e, args.num_epochs, i+1, len(test_loader), loss, total_loss, acc, total_acc)
        m, h = mean_confidence_interval(test_acc, confidence=0.95)
        print(" {:.2f}+-{:.2f}".format(m, h))