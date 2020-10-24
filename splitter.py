import os
import glob
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--frames-path", type=str, default="../Data/UCF101/UCF101_frames/")
parser.add_argument("--labels-path", type=str, default="../Data/UCF101/UCF101_labels/")
parser.add_argument("--save-path", type=str, default="./UCF101_few_shot_labels/")
parser.add_argument("--number-of-train", type=int, default=71)
parser.add_argument("--number-of-test", type=int, default=30)
args = parser.parse_args()

# check directory
assert os.path.exists(args.save_path) == False, "'{}' directory is alreay exist !!".format(args.frames_path)
os.makedirs(args.save_path)

with open(os.path.join(args.labels_path, "classInd.txt")) as f:
    categories = f.readlines()

categories = np.random.permutation(categories)

directory_name = []
for d in glob.glob(args.frames_path + "*"):
    if os.path.isdir(d):
        directory_name.append(d.split("\\" if os.name == 'nt' else "/")[-1])
directory_name = pd.DataFrame(directory_name)

# save train labels
with open(os.path.join(args.save_path, "train.csv"), 'w') as f:
    first = True
    for i, c in enumerate(categories[:args.number_of_train]):
        print("writing {}===".format(c))
        lines = np.concatenate(directory_name.loc[directory_name[0].str.contains('_' + c.strip('\n').split(' ')[1] + '_')].values, axis=0)
        for line in lines:
            f.write(str(i+1) + ',' + line) if first else f.write('\n' + str(i+1) + ',' + line)
            first = False

# save test labels
with open(os.path.join(args.save_path, "test.csv"), 'w') as f:
    first = True
    for i, c in enumerate(categories[args.number_of_train:args.number_of_train+args.number_of_val]):
        print("writing {}===".format(c))
        lines = np.concatenate(directory_name.loc[directory_name[0].str.contains('_' + c.strip('\n').split(' ')[1] + '_')].values, axis=0)
        for line in lines:
            f.write(str(i+1) + ',' + line) if first else f.write('\n' + str(i+1) + ',' + line)
            first = False