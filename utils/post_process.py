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
