import os
from glob import glob

frames_path = "../Data/UCF101/UCF101_frames/"
hit_sequence_length = 35

frames_path_list = glob(os.path.join(frames_path, "*"))

# check total videos(from frames path)
print("total videos: {}".format(len(frames_path_list)))

counter = 0
for frame_path in frames_path_list:
    sequence_length = len(glob(os.path.join(frame_path, "*")))
    if hit_sequence_length > sequence_length:
        print("hit ! {}, sequence length: {}".format(frame_path, sequence_length))
        counter += 1

print("total {} videos has short sequence rather than {}".format(counter, hit_sequence_length))