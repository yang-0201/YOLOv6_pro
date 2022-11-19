# YOLOv6_pro
Make it easier for yolov6 to change the network structure

Already supported models:
YOLOV6l,
YOLOV6m,
YOLOV6t,
YOLOV6s
<summary> 数据集配置 </summary>

```
images/train 中放入你的训练集图片
images/val 中放入你的验证集图片
labels/train 中放入你的训练集标签
labels/val 中放入你的验证集标签 
```
<summary> 数据集文件结构 </summary>

```
├── data
│   ├── images
│   │   ├── train
│   │   └── val
│   ├── labels
│   │   ├── train
│   │   ├── val
```

<summary> data.yaml配置 </summary>

```shell
train: data/images/train # train images
val: data/images/val # val images
```

<summary> 训练命令 </summary>

```shell
python tools/train.py --batch 8 --conf configs/model_yaml/yolov6t_yaml.py --data data/car.yaml --device 0 --img 640 --epochs 100
```
