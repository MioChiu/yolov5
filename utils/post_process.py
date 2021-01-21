import os
import cv2
import torch
import json
import numpy as np
from ensemble_boxes import *

def get_tile_edge(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    b = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_image = b[1]
    # print(binary_image.shape)
    h = binary_image.shape[0]
    w = binary_image.shape[1]

    edges_y = []
    edges_x = []
    for i in range(0, h, 50):
        for j in range(0, w, 50):
            if binary_image[i][j] == 255:
                edges_y.append(i)
                edges_x.append(j)

    left = max(min(edges_x) - 100, 0) 
    right = min(max(edges_x) + 100, w)
    # width = right - left
    bottom = max(min(edges_y) - 100, 0)
    top = min(max(edges_y) + 100, h)
    # height = top - bottom

    # pre1_picture = image[bottom:bottom+height, left:left+width]

    return left, bottom, right, top

def soft_nms_pytorch(dets, box_scores, sigma=0.5, thresh=0.001):

    """
    reference https://github.com/DocF/Soft-NMS.git
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        boxes coordinate tensor (format:[x1, y1, x2, y2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        thresh:      score thresh       
    # Return
        the sorted index of the selected boxes
    """
    N = dets.shape[0]  # the number of boxes

    # Indexes concatenate boxes with the last column
    indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1) 
    dets = torch.cat((dets, indexes), dim=1)

    # Sort the scores of the boxes from largest to smallest
    box_scores, conf_sort_index = torch.sort(box_scores, descending=True)
    dets = dets[conf_sort_index]

    for i in range(N):

        pos=i+1

        #iou calculate
        ious = box_iou(dets[i][0:4].view(-1,4), dets[pos:,:4])
        

        # Gaussian decay 
        box_scores[pos:] = torch.exp(-(ious * ious) / sigma) * box_scores[pos:]

        box_scores[pos:] = box_scores[pos:]
        box_scores[pos:], arg_sort = torch.sort(box_scores[pos:], descending=True)

        a=dets[pos:]
        
        dets[pos:] = a[arg_sort]

     # select the boxes and keep the corresponding indexes
    keep = dets[:,4][box_scores>thresh].long()

    return keep

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def run_wbf(boxes, scores, labels, image_size, iou_thr=0.3, skip_box_thr=0.01, weights=None):
    #boxes = [prediction[image_index]['boxes'].data.cpu().numpy()/(image_size-1) for prediction in predictions]
    #scores = [prediction[image_index]['scores'].data.cpu().numpy() for prediction in predictions]
    # labels = [np.zeros(score.shape[0]) for score in scores]
    image_size = np.array([image_size[1], image_size[0], image_size[1], image_size[0]])
    boxes = [box/(image_size) for box in boxes]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    #boxes, scores, labels = nms(boxes, scores, labels, weights=[1,1,1,1,1], iou_thr=0.5)
    boxes = boxes*(image_size)
    # mask = scores > 0.05
    # boxes = boxes[mask]
    # scores = scores[mask]
    # labels = labels[mask]
    scores = scores[:, np.newaxis]
    labels = labels[:, np.newaxis]
    return boxes, scores, labels