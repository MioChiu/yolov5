import os
import cv2
import torch
import json
import numpy as np

def merge(pred):
    """
     pred:[batch_size x [n x 6]]
    """
    cat_pred = np.array([])
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] # normalization gain whwh
    for i, det in enumerate(pred):  # detections per image
        # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
        s += '%gx%g ' % patch_images[i].shape[1:]  # print string
            
        if len(det):
            # Rescale boxes from img_size to patch img size
            det[:, :4] = scale_coords(patch_images[i].shape[1:], det[:, :4], patch_size).round()
            # convert coord to img0 coordinate system
            origin = torch.tensor([patch_borders[i][0], patch_borders[i][1], patch_borders[i][0], patch_borders[i][1]]).to(device)
            det[:, :4] = det[:, :4] + origin
            if len(cat_pred):
                cat_pred = np.concatenate([cat_pred, det], 0)
            else:
                cat_pred = det
    for c in classes:
        mask = cat_pred[:, 5] == c
        cls_pred = cat_pred[mask]
        _, order = torch.sort(cls_pred[:, 5], descending=True)

        keep = []
        while order.numel() > 0:       # torch.numel()返回张量元素个数
            if order.numel() == 1:     # 保留框只剩一个
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()    # 保留scores最大的那个框box[i]
                keep.append(i)

            iou = box_iou(cls_pred[i], cls_pred[order[1:]])
            idx = (iou <= threshold).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]
            if idx.numel() == 0:
                break
            order = order[idx+1]  # 修补索引之间的差值
        return torch.LongTensor(keep)   # Pytorch的索引值为LongTensor


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