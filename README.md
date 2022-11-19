# YOLOv6_pro
Make it easier for yolov6 to change the network structure

Already supported models:
YOLOV6l,
YOLOV6m,
YOLOV6t,
YOLOV6s
<summary> 数据集配置 </summary>

```
data/images/train 中放入你的训练集图片
data/images/val 中放入你的验证集图片
data/labels/train 中放入你的训练集标签(标签格式为yolo格式)
data/labels/val 中放入你的验证集标签 
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

<summary> data.yaml 配置 </summary>

```shell
train: data/images/train # train images
val: data/images/val # val images
is_coco: False
nc: 3  # number of classes
names: ["car","person","bike"] #classes names
```

<summary> 预训练权重放在 weights 目录 </summary>



<summary> 训练命令 </summary>

```shell
python tools/train.py --conf configs/model_yaml/yolov6t_yaml.py --data data/car.yaml --device 0 --img 640
```

```shell
python tools/train.py --conf configs/model_yaml/yolov6s_yaml.py --data data/car.yaml --device 0 --img 640
```

```shell
python tools/train.py --conf configs/model_yaml/yolov6m_yaml.py --data data/car.yaml --device 0 --img 640
```

```shell
python tools/train.py --conf configs/model_yaml/yolov6l_yaml.py --data data/car.yaml --device 0 --img 640
```
