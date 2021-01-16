import argparse
import time
from pathlib import Path
from numpy import random

import os
import cv2
import torch
import json
import torch.backends.cudnn as cudnn
import numpy as np


from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadTileImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.process_test_tile_data import merge_sub_det
from utils.post_process import get_tile_edge

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    patch_size = opt.patch_size
    save_json = opt.save_json
    # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    #     ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    # classify = False
    # if classify:
    #     modelc = load_classifier(name='resnet101', n=2)  # initialize
    #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    save_img = True
    dataset = LoadTileImages(source, img_size=imgsz, patch_size=patch_size)  # will convert image to patch 

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    # json for eval
    results_json = []
    json_path = str(save_dir / 'results') + '.json'

    for path, patch_images, im0s, patch_borders in dataset:
        patch_images = np.asarray(patch_images)
        patch_images = torch.from_numpy(patch_images).to(device)
        patch_images = patch_images.half() if half else patch_images.float()  # uint8 to fp16/32
        patch_images /= 255.0  # 0 - 255 to 0.0 - 1.0
        if patch_images.ndimension() == 3:
            patch_images = patch_images.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(patch_images, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        
        # Apply Classifier
        # if classify:
        #     pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        sub_boxes_list = []
        sub_scores_list = []
        sub_labels_list = []
        cat_pred = np.array([])

        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] # normalization gain whwh
        for i, det in enumerate(pred):  # detections per image
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % patch_images[i].shape[1:]  # print string
              
            if len(det):
                # Rescale boxes from img_size to patch img size
                det[:, :4] = scale_coords(patch_images[i].shape[1:], det[:, :4], patch_size).round()
                # det = det.cpu().numpy()
                # convert coord to img0 coordinate system
                origin = np.array([patch_borders[i][0], patch_borders[i][1], patch_borders[i][0], patch_borders[i][1]])
                sub_boxes = (det[:, :4].cpu() + origin) / gn 
                # det[:, :4] = det[:, :4] + origin
                # if len(cat_pred):
                #     cat_pred = np.concatenate([cat_pred, det], 0)
                # else:
                #     cat_pred = det
                sub_boxes_list.append(sub_boxes.numpy())
                sub_scores_list.append(det[:, 4].cpu().numpy())
                sub_labels_list.append(det[:, 5].cpu().numpy())
        # merge all subimg 
        boxes, scores, classes = merge_sub_det(sub_boxes_list, sub_scores_list, sub_labels_list)
        boxes = boxes * gn.numpy()
        # boxes, scores, classes = cat_pred[:, :4], cat_pred[:, 4], cat_pred[:, 5]
        t2 = time_synchronized()
        print(f'Done. ({t2 - t1:.3f}s)')

        # discard background
        ex1, ey1, ex2, ey2 = get_tile_edge(im0)
        discard = (ex1, ey1, ex2, ey2)
        if save_img or view_img:  # Add bbox to image
            for i in range(len(boxes)):
                xyxy = boxes[i]
                if xyxy[0] < discard[0] or xyxy[1] < discard[1] or xyxy[2] > discard[2] or xyxy[3] > discard[3]:
                    continue
                conf = scores[i]
                cls_ = classes[i]
                label = f'{names[int(cls_)]} {conf:.2f}'
                # plot_one_box(xyxy, im0, label=label, color=colors[int(cls_)], line_thickness=3)
                results_dict = {
                    "name": p.name,
                    "category": int(cls_) + 1,
                    "bbox": list(xyxy.astype(np.float64)),
                    "score": np.float64(conf)
                }
                results_json.append(results_dict)
        # cv2.imwrite(save_path, im0)
    if save_json:
        with open(json_path, 'w') as f:
            json.dump(results_json, f)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/mnt/qiuzheng/codes/exp_yolov5/yolov5/runs/train/5x_se_e200_832/weights/best74.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='/mnt/qiuzheng/data/tile/images/test/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=832, help='inference size (pixels)')
    parser.add_argument('--patch-size', nargs='+', type=int, default=[832, 832], help='[train, test] image sizes')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-json', default=True, help='save results to *.json')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', default=True, help='augmented inference')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='5x_bce_e300_832_new_74_test', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
