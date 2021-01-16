import yaml

path = '/mnt/qiuzheng/codes/exp_yolov5/yolov5/models/yolov5x_dcn.yaml'
with open(path, 'r') as f:
    yaml_ = yaml.load(f, Loader=yaml.FullLoader) 
layer_nams = []
for i, (f, n, m, args) in enumerate(yaml_['backbone'] + yaml_['head']):
    if m == 'DCN':
        print('1')
        break