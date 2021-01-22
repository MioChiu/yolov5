import os
import json
import math
import shutil
import numpy as np

if __name__ == "__main__":
    cls_dict = {
        "0": "背景",
        "1": "边异常",
        "2": "角异常",
        "3": "白色点瑕疵",
        "4": "浅色块瑕疵",
        "5": "深色点块瑕疵",
        "6": "光圈瑕疵"
    }

    train_json_path = '/mnt/qiuzheng/data/tc/tile_round1_train_20201231/train_annos.json'
    save_txt_dir = '/mnt/qiuzheng/data/tile/labels/train'

    src_trian_img_dir = '/mnt/qiuzheng/data/tc/tile_round1_train_20201231/train_imgs'
    dst_trian_img_dir = '/mnt/qiuzheng/data/tile/images/train'
    src_test_img_dir = '/mnt/qiuzheng/data/tc/tile_round1_testA_20201231/testA_imgs'
    dst_test_img_dir = '/mnt/qiuzheng/data/tile/images/test'

    # create dst dirs
    if not os.path.exists(save_txt_dir):
        os.makedirs(save_txt_dir)
    # if not os.path.exists(dst_trian_img_dir):
    #     os.makedirs(dst_trian_img_dir)
    # if not os.path.exists(dst_test_img_dir):
    #     os.makedirs(dst_test_img_dir)

    # read json
    with open(train_json_path, 'r', encoding='utf8') as fp:
        data_list = json.load(fp)

    # process data and convert json to txt
    # yolo data format [cls, cx, cy, w, h]
    for data in data_list:
        jpg_name = data["name"]
        category = str(data["category"] - 1)
        x1, y1, x2, y2 = data["bbox"]
        height = data["image_height"]
        width = data["image_width"]

        w = x2 - x1
        h = y2 - y1
        center_x = str((x1 + w / 2) / width)
        center_y = str((y1 + h / 2) / height)
        w = str(w / width)
        h = str(h / height)
        line = ' '.join([category, center_x, center_y, w, h])

        txt_name = jpg_name.replace('jpg', 'txt')
        txt_path = os.path.join(save_txt_dir, txt_name)
        with open(txt_path, 'a') as f:
            f.write(line + '\n')

    # copy img to dst dirs
    shutil.copytree(src_trian_img_dir, dst_trian_img_dir)
    shutil.copytree(src_test_img_dir, dst_test_img_dir)