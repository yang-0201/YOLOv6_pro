# YOLOv6_pro
Make it easier for yolov6 to change the network structure.<br>
Build a YOLOv6 network using the YOLOv5 style of building networks, modifying your modules in the yaml file as you wish.<br>
Already supported models: `YOLOV6l`, `YOLOV6m`, `YOLOV6t`, `YOLOV6s`

## 数据集配置
```
data/images/train 中放入你的训练集图片
data/images/val 中放入你的验证集图片
data/labels/train 中放入你的训练集标签(标签格式为yolo格式)
data/labels/val 中放入你的验证集标签 
```
### 数据集文件结构
```
├── data
│   ├── images
│   │   ├── train
│   │   └── val
│   ├── labels
│   │   ├── train
│   │   ├── val
```
### data.yaml 配置
```shell
train: data/images/train # train images
val: data/images/val # val images
is_coco: False
nc: 3  # number of classes
names: ["car","person","bike"] #classes names
```
## 预训练权重
  [YOLOv6l_yaml.pt](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6l_yaml.pt)<br>
  [YOLOv6m_yaml.pt](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6m_yaml.pt)<br>
  [YOLOv6s_yaml.pt](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6s_yaml.pt)<br>
  [YOLOv6t_yaml.pt](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6t_yaml.pt)<br>
## 训练命令
YOLOv6t
```shell
python tools/train.py --conf configs/model_yaml/yolov6t_yaml.py --data data/car.yaml --device 0 --img 640
```
YOLOv6s
```shell
python tools/train.py --conf configs/model_yaml/yolov6s_yaml.py --data data/car.yaml --device 0 --img 640
```
YOLOv6m
```shell
python tools/train.py --conf configs/model_yaml/yolov6m_yaml.py --data data/car.yaml --device 0 --img 640
```
YOLOv6l
```shell
python tools/train.py --conf configs/model_yaml/yolov6l_yaml.py --data data/car.yaml --device 0 --img 640
```
