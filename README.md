## 简介
本项目基于[yolov5](https://github.com/ultralytics/yolov5)。

## 环境
### requirements
```bash
$ pip install -r requirements.txt
```
### 编译DCN(可选)
```bash
$ cd code/models/DCNv2
$ ./make.sh
```
### 安装wbf
```bash
$ cd code/Weighted-Boxes-Fusion
$ pip install ensemble-boxes
```
## Tutorials
### 准备数据集
将原始的数据集转化为yolo格式,并划分训练集验证集：
```bash
$ cd ./code
$ python scripts/data2yolo.py
$ python scripts/slice_data_yolo.py
```

由于图片尺寸过大，直接用原图大小训练显存不够用，用滑窗将原图割成若干832*832的子图，保存包含gt的图片：
```bash
$ python scripts/cut_images.py
```

### 训练
单卡：
```bash
$ python train.py --weights 'weights/yolov5x.pt' --cfg models/yolov5x_se.yaml --data data/tile.yaml --epoch 150 --batch-size 32 --img-size 832 --device 0 --name 5x_se_e150_s832
```
多卡DP：
```bash
$ python train.py --weights 'weights/yolov5x.pt' --cfg models/yolov5x_se.yaml --data data/tile.yaml --epoch 150 --batch-size 32 --img-size 832 --device 0,1 --name 5x_se_e150_s832
```
多卡DDP:
```bash
$ python -m torch.distributed.launch --nproc_per_node 2 train.py --weights 'weights/yolov5x.pt' --cfg models/yolov5x_se.yaml --data data/tile.yaml --epoch 150 --batch-size 32 --img-size 832 --device 0,1 --name 5x_se_e150_s832
```

训练完成后复制模型到指定的目录：
```bash
$ cp runs/train/5x_se_e150_s832/weights/best.pt ../user_data/model_data/best.pt
```

### 测试并生成result.json
```bash
$ python detect_tile_whole.py
```