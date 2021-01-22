import json
import os
import shutil
import random
import json

ann_root = '/mnt/sjk/data/traindata/bag/annotations'
img_root = '/mnt/sjk/data/traindata/bag/images'
img_list = []
ann_list = []
miss_num = 0

for home, dirs, files in os.walk(ann_root):
    for filename in files:
        if 'json' in filename:
            ann_path = os.path.join(home, filename)
            img_path = ann_path.replace('annotations', 'images').replace('json', 'jpg')
            if not os.path.exists(img_path):
                miss_num += 1
                print(ann_path, img_path)
                continue
            img_list.append(img_path)
random.shuffle(img_list)
train_list = img_list[:int(len(img_list) * 0.99)]
val_list = img_list[int(len(img_list) * 0.99):]

print(len(train_list))
print(len(val_list))
print(miss_num)

split = {"train": train_list, "val": val_list}
json.dump(split, open("/mnt/qiuzheng/data/det/bag/split.json", "w", encoding="utf-8"),
          ensure_ascii=False)
