import os
import glob
import random
import fnmatch
import re
import sys

def get_list(path, level=1):
    if level == 1:
        folders = glob.glob(os.path.join(path, '*'))
    elif level == 2:
        folders = glob.glob(os.path.join(path, '*', '*'))
    else:
        raise ValueError('level can be only 1 or 2')

    list_folders = [(folder + ' ' + folder.split("0")[0].split(os.sep)[-1]) for folder in folders]

    return list_folders


def fight_splits(video_list):
    videos = video_list

    print("videos:", videos[0])

    with open("gallery.txt", "w") as f:
        for item in videos:
            f.write(item + "\n")


if __name__ == "__main__":
    frame_dir = sys.argv[1]  # "rawframes"
    level = int(sys.argv[2])  # 2

    video_list = get_list(frame_dir, level=level)
    print("number:", len(video_list))
    print("video_list:", video_list)

    fight_splits(video_list)
