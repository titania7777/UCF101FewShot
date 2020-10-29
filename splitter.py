import os
import glob
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--frames-path", type=str, default="../UCF101FrameExtractor/UCF101_frames/")
parser.add_argument("--save-path", type=str, default="./UCF101_few_shot_labels/")
parser.add_argument("--categories", type=str, default="./categories.txt")
parser.add_argument("--number-of-train", type=int, default=71)
parser.add_argument("--number-of-test", type=int, default=30)
args = parser.parse_args()

# check frames path
assert os.path.exists(args.frames_path), "'{}' path does not exist.".format(args.frames_path)

# check save directory
assert os.path.exists(args.save_path) == False, "'{}' directory is alreay exist !!".format(args.frames_path)
os.makedirs(args.save_path)

with open(args.categories) as f:
    categories = f.read().splitlines() 

categories = np.random.permutation(categories)

video_names = []
for d in glob.glob(args.frames_path + "*"):
    if os.path.isdir(d):
        video_names.append(d.split("\\" if os.name == 'nt' else "/")[-1])
video_names = pd.DataFrame(video_names)

# save train labels
with open(os.path.join(args.save_path, "train.csv"), 'w') as f:
    first = True
    for i, c in enumerate(categories[:args.number_of_train]):
        print("writing... {} ".format(c))
        lines = np.concatenate(video_names[video_names[0].str.contains("_" + c + "_")].values.tolist(), axis=0)
        for line in lines:
            f.write(str(i+1) + ',' + line) if first else f.write('\n' + str(i+1) + ',' + line)
            first = False

# save test labels
with open(os.path.join(args.save_path, "test.csv"), 'w') as f:
    first = True
    for i, c in enumerate(categories[args.number_of_train:]):
        print("writing... {} ".format(c))
        lines = np.concatenate(video_names[video_names[0].str.contains("_" + c + "_")].values.tolist(), axis=0)
        for line in lines:
            f.write(str(i+1) + ',' + line) if first else f.write('\n' + str(i+1) + ',' + line)
            first = False