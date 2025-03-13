import rerun as rr
import numpy as np
from pathlib import Path
import argparse
import yaml

def read_kitti_file(path):
    # Unsure about dtype and about shape.
    # Are we sure that the points have the shape (x, y, z, intensity)?
    return np.fromfile(path, dtype='<f4').reshape(-1, 4)

def same_length(d):
    first_length = len(next(iter(d.values())))

    for v in d.values():
        if len(v) != first_length:
            raise ValueError(f"All values must have the same length. Found a value with length {len(v)}.")
    return first_length

def visualize(f):
    n = same_length(f)

    rr.init("PCD and image sequence", spawn=True)
    for i in range (n):
        rr.set_time_sequence("frame", i)
        print(f"Timestamp: {i}")
        for k, v in f.items():
            print(f"{k}: {v[i].name}")
            pc = read_kitti_file(v[i])
            rr.log(k, rr.Points3D(pc[:, 0:3]))

def parse_arguments():
    parser = argparse.ArgumentParser(description='LiDAR foggification')
    parser.add_argument("-c", "--config", help="path to config file", type=str)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    return config


def main(config):
    files_dict = config["data"]
    for k, v in files_dict.items():
        files_dict[k] = sorted(list(Path(v).glob("*.bin")))
    visualize(files_dict)

if __name__ == '__main__':
    config = parse_arguments()
    main(config)