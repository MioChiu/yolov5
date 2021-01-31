import argparse
import numpy as np
import torch

from utils.general import strip_optimizer


if __name__ == '__main__':

    src_path = '/mnt/qiuzheng/codes/exp_yolov5/yolov5/runs/train/5x_se_lrf04_box025_a4_t5_e150_832/weights/best93.pt'
    save_path = '/mnt/qiuzheng/codes/exp_yolov5/yolov5/runs/train/5x_se_lrf04_box025_a4_t5_e150_832/weights/best93_half.pt'
    strip_optimizer(src_path, save_path)
