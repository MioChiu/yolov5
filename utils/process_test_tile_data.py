import os
import cv2
import numpy as np
from utils.ensemble_boxes_wbf import weighted_boxes_fusion


def get_patch_images(img, height=1024, width=1024, overlap=0.125):
    """
    get patch images when testing
    """
    # calculate stride
    # stride_h=896, stride_w=896
    stride_h = height - int(height * overlap)
    stride_w = width - int(width * overlap)

    # read img
    # img = cv2.imread(img_path)

    img_h, img_w, _ = img.shape
    assert (img_h, img_w) in [(3500, 4096), (6000, 8192)]

    patch_images = []
    patch_borders = []

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
            sub_img = img[top_left_row:bottom_right_row, top_left_col: bottom_right_col]

            patch_images.append(sub_img)
            patch_borders.append([top_left_col, top_left_row, bottom_right_col, bottom_right_row])
    return patch_images, patch_borders


def merge_sub_det(sub_boxes_list, sub_scores_list, sub_labels_list):
    """
    sub_det_list : a list of all subimg results. [(n1, 6), (n2, 6) ...] x1 y1 x2 y2 score cls 
    patch_borders : x1y1x2y2 of subimg border
    """
    # sub_det_list中x1y1x2y2为目标在子图上的坐标 比如子图的size(1024,1024),那么就是这个(1024,1024)坐标系上的，而不是model输入图片的size(832*832?)上的。
    # assert len(sub_boxes_list) == len(patch_borders)
    # for i in range(len(sub_boxes_list)):
    #     border = np.array(patch_borders[i])
    #     sub_boxes_list[i] += border

    iou_thr = 0.2
    skip_box_thr = 0.0001
    boxes, scores, labels = weighted_boxes_fusion(sub_boxes_list, sub_scores_list, sub_labels_list, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    return boxes, scores, labels