import os
import cv2
import random
import numpy as np
from PIL import Image


def save_patch(patch_images, patch_boxes, patch_classes, img_dir, lb_dir, name):
    for i in range(len(patch_images)):
        savename = name + '_{}'.format(i)
        img_path = os.path.join(img_dir, savename + '.jpg')
        lb_path = os.path.join(lb_dir, savename + '.txt')
        # cv2.imwrite(img_path, patch_images[i])
        patch_images[i].save(img_path)
        assert len(patch_boxes[i]) == len(patch_classes[i])
        with open(lb_path, 'a') as f:
            for j in range(len(patch_boxes[i])):
                category = str(patch_classes[i][j])
                cx = str(patch_boxes[i][j][0])
                cy = str(patch_boxes[i][j][1])
                w = str(patch_boxes[i][j][2])
                h = str(patch_boxes[i][j][3])
                line = ' '.join([category, cx, cy, w, h])
                f.write(line + '\n')
    # for i in range(len(neg_patch_images)):
    #     savename = name + '_neg_{}'.format(i)
    #     img_path = os.path.join(img_dir, savename + '.jpg')
    #     lb_path = os.path.join(lb_dir, savename + '.txt')
    #     cv2.imwrite(img_path, neg_patch_images[i])
    #     with open(lb_path, 'w') as f:
    #         pass


def cal_overlap(boxes, bg):
    """
    boxes:[n, 4] x1y1x2y2
    bg:[4,] x1y1x2y2
    """
    x1 = np.maximum(boxes[:, 0], bg[0])
    y1 = np.maximum(boxes[:, 1], bg[1])
    x2 = np.minimum(boxes[:, 2], bg[2])
    y2 = np.minimum(boxes[:, 3], bg[3])

    in_h = np.maximum(y2 - y1, 0)
    in_w = np.maximum(x2 - x1, 0)

    inter = in_h * in_w

    return inter


def cut_one_img(img_path, lb_path=None, height=1024, width=1024, overlap=0.125):
    # calculate stride
    # stride_h=896, stride_w=896
    stride_h = height - int(height * overlap)
    stride_w = width - int(width * overlap)

    # read img
    # img = cv2.imread(img_path)
    img = Image.open(img_path)
    img_w, img_h = img.size
    assert (img_h, img_w) in [(3500, 4096), (6000, 8192)]

    # read ann
    with open(lb_path) as f:
        lines = f.readlines()
    classes = []
    boxes = []
    for line in lines:
        # yolo data format [cls, cx, cy, w, h], coordinate is normalized.
        line = line.strip().split(' ')
        category = int(line[0])
        # normalized coord => original coord
        obj_cx = img_w * float(line[1])
        obj_cy = img_h * float(line[2])
        obj_w = img_w * float(line[3])
        obj_h = img_h * float(line[4])
        # cx cx w h => x1 y1 x2 y2
        obj_x1 = obj_cx - obj_w / 2
        obj_y1 = obj_cy - obj_h / 2
        obj_x2 = obj_cx + obj_w / 2
        obj_y2 = obj_cy + obj_h / 2
        classes.append(category)
        boxes.append([obj_x1, obj_y1, obj_x2, obj_y2])
    classes = np.asarray(classes)  # (n,)
    boxes = np.asarray(boxes)  # (n, 4)

    patch_images = []
    patch_boxes = []
    patch_classes = []
    # neg_patch_images = []
    for start_h in range(0, img_h, stride_h):
        for start_w in range(0, img_w, stride_w):
            if start_h + height > img_h:
                start_h = img_h - height
            if start_w + width > img_w:
                start_w = img_w - width
            top_left_row = max(start_h, 0)
            top_left_col = max(start_w, 0)
            bottom_right_row = min(start_h + height, img_h)
            bottom_right_col = min(start_w + width, img_w)
            assert (bottom_right_row - top_left_row, bottom_right_col - top_left_col) == (height, width)
            # sub_img = img[top_left_row:bottom_right_row, top_left_col: bottom_right_col]
            sub_img = img.crop((top_left_col, top_left_row, bottom_right_col, bottom_right_row))

            # calculate new coord in subimg
            new_boxes = np.zeros_like(boxes)
            new_boxes[:, 0] = boxes[:, 0] - top_left_col
            new_boxes[:, 2] = boxes[:, 2] - top_left_col
            new_boxes[:, 1] = boxes[:, 1] - top_left_row
            new_boxes[:, 3] = boxes[:, 3] - top_left_row

            inter = cal_overlap(new_boxes, [0, 0, width, height])
            idx = np.where(inter > 0)[0]

            if len(idx) > 0:
                sub_boxes = new_boxes[idx, :]
                sub_classes = classes[idx]
                # slice box
                sub_boxes[:, 0] = np.minimum(np.maximum(sub_boxes[:, 0], 0), width)
                sub_boxes[:, 1] = np.minimum(np.maximum(sub_boxes[:, 1], 0), height)
                sub_boxes[:, 2] = np.minimum(np.maximum(sub_boxes[:, 2], 0), width)
                sub_boxes[:, 3] = np.minimum(np.maximum(sub_boxes[:, 3], 0), height)
                # x1 y1 x2 y2 => cx cx w h && normalize
                sub_w = (sub_boxes[:, 2] - sub_boxes[:, 0]) / width
                sub_h = (sub_boxes[:, 3] - sub_boxes[:, 1]) / height
                sub_cx = 0.5 * (sub_boxes[:, 0] + sub_boxes[:, 2]) / width
                sub_cy = 0.5 * (sub_boxes[:, 1] + sub_boxes[:, 3]) / height
                sub_boxes = np.concatenate([sub_cx[:, np.newaxis], sub_cy[:, np.newaxis], sub_w[:, np.newaxis], sub_h[:, np.newaxis]], 1)

                patch_images.append(sub_img)
                patch_boxes.append(sub_boxes)
                patch_classes.append(sub_classes)
            # 放一些负样本?
            # else:
            #     if random.random() <= 0.02:
            #         neg_patch_images.append(sub_img)
    return patch_images, patch_boxes, patch_classes  # , neg_patch_images


def cut_one_img_random_step(img_path, lb_path=None, height=1024, width=1024, overlap=0.125):
    # calculate stride
    # stride_h=896, stride_w=896
    stride_h = height - int(height * overlap)
    stride_w = width - int(width * overlap)

    # read img
    # img = cv2.imread(img_path)
    img = Image.open(img_path)
    img_w, img_h = img.size
    assert (img_h, img_w) in [(3500, 4096), (6000, 8192)]

    # read ann
    with open(lb_path) as f:
        lines = f.readlines()
    classes = []
    boxes = []
    for line in lines:
        # yolo data format [cls, cx, cy, w, h], coordinate is normalized.
        line = line.strip().split(' ')
        category = int(line[0])
        # normalized coord => original coord
        obj_cx = img_w * float(line[1])
        obj_cy = img_h * float(line[2])
        obj_w = img_w * float(line[3])
        obj_h = img_h * float(line[4])
        # cx cx w h => x1 y1 x2 y2
        obj_x1 = obj_cx - obj_w / 2
        obj_y1 = obj_cy - obj_h / 2
        obj_x2 = obj_cx + obj_w / 2
        obj_y2 = obj_cy + obj_h / 2
        classes.append(category)
        boxes.append([obj_x1, obj_y1, obj_x2, obj_y2])
    classes = np.asarray(classes)  # (n,)
    boxes = np.asarray(boxes)  # (n, 4)

    patch_images = []
    patch_boxes = []
    patch_classes = []
    # neg_patch_images = []

    start_h = random.randint(0, 32)
    while start_h + height <= img_h:
        start_w = random.randint(16, 128)
        while start_w + width <= img_w:
            top_left_row = max(start_h, 0)
            top_left_col = max(start_w, 0)
            bottom_right_row = min(start_h + height, img_h)
            bottom_right_col = min(start_w + width, img_w)
            assert (bottom_right_row - top_left_row, bottom_right_col - top_left_col) == (height, width)
            # sub_img = img[top_left_row:bottom_right_row, top_left_col: bottom_right_col]
            sub_img = img.crop((top_left_col, top_left_row, bottom_right_col, bottom_right_row))

            # calculate new coord in subimg
            new_boxes = np.zeros_like(boxes)
            new_boxes[:, 0] = boxes[:, 0] - top_left_col
            new_boxes[:, 2] = boxes[:, 2] - top_left_col
            new_boxes[:, 1] = boxes[:, 1] - top_left_row
            new_boxes[:, 3] = boxes[:, 3] - top_left_row

            inter = cal_overlap(new_boxes, [0, 0, width, height])
            idx = np.where(inter > 0)[0]

            if len(idx) > 0:
                sub_boxes = new_boxes[idx, :]
                sub_classes = classes[idx]
                # slice box
                sub_boxes[:, 0] = np.minimum(np.maximum(sub_boxes[:, 0], 0), width)
                sub_boxes[:, 1] = np.minimum(np.maximum(sub_boxes[:, 1], 0), height)
                sub_boxes[:, 2] = np.minimum(np.maximum(sub_boxes[:, 2], 0), width)
                sub_boxes[:, 3] = np.minimum(np.maximum(sub_boxes[:, 3], 0), height)
                # x1 y1 x2 y2 => cx cx w h && normalize
                sub_w = (sub_boxes[:, 2] - sub_boxes[:, 0]) / width
                sub_h = (sub_boxes[:, 3] - sub_boxes[:, 1]) / height
                sub_cx = 0.5 * (sub_boxes[:, 0] + sub_boxes[:, 2]) / width
                sub_cy = 0.5 * (sub_boxes[:, 1] + sub_boxes[:, 3]) / height
                sub_boxes = np.concatenate([sub_cx[:, np.newaxis], sub_cy[:, np.newaxis], sub_w[:, np.newaxis], sub_h[:, np.newaxis]], 1)

                patch_images.append(sub_img)
                patch_boxes.append(sub_boxes)
                patch_classes.append(sub_classes)
            
            start_w += width - random.randint(64, 320)
        start_h += height - random.randint(64, 320)
    return patch_images, patch_boxes, patch_classes  # , neg_patch_images


def patch_generator(src_img_dir, src_lb_dir, sub_img_dir, sub_lb_dir):
    if not os.path.exists(sub_img_dir):
        os.makedirs(sub_img_dir)
    if not os.path.exists(sub_lb_dir):
        os.makedirs(sub_lb_dir)
    img_names = os.listdir(src_img_dir)
    num_pos = 0
    # num_neg = 0
    for img_name in img_names:
        name = img_name.replace('.jpg', '')
        img_path = os.path.join(src_img_dir, img_name)
        lb_path = os.path.join(src_lb_dir, name + '.txt')
        print(img_name)
        patch_images, patch_boxes, patch_classes = cut_one_img(img_path, lb_path, height=832, width=832, overlap=0.077)
        num_pos += len(patch_images)
        # num_neg += len(neg_patch_images)
        save_patch(patch_images, patch_boxes, patch_classes, sub_img_dir, sub_lb_dir, name)


def _unit_test():
    img_path = '254_99_t20201130153406261_CAM1.jpg'
    lb_path = '254_99_t20201130153406261_CAM1.txt'
    name = '254_99_t20201130153406261_CAM1'
    patch_images, patch_boxes, patch_classes = cut_one_img(img_path, lb_path, height=832, width=832, overlap=0.077)
    save_patch(patch_images, patch_boxes, patch_classes, './img', './lb', name)


if __name__ == "__main__":
    src_img_dir = '/mnt/qiuzheng/data/tile/images/train'
    src_lb_dir = '/mnt/qiuzheng/data/tile/labels/train'
    sub_img_dir = '/mnt/qiuzheng/data/tile_832_whole/images/train'
    sub_lb_dir = '/mnt/qiuzheng/data/tile_832_whole/labels/train'
    patch_generator(src_img_dir, src_lb_dir, sub_img_dir, sub_lb_dir)
