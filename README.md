# YOLOv6_pro
让 `YOLOv6` 更换网络结构更为方便 <br><br>
基于官方 `YOLOv6` 的整体架构，使用 `YOLOv5` 的网络构建方式构建一个 `YOLOv6` 网络，包括 `backbone`，`neck`，`effidehead` 结构 <br><br>
可以在 `yaml` 文件中任意修改或添加模块,并且每个修改的文件都是独立可运行的 <br><br>
预训练权重已经从官方权重转换，确保可以匹配 <br>
## 已经支持的模型:
<li>YOLOV6l_yaml</li>
<li>YOLOV6m_yaml</li>
<li>YOLOV6s_yaml</li>
<li>YOLOV6t_yaml</li><br>
大尺寸模型，四个输出层：
<li>YOLOV6l6_p2_yaml</li>
<li>YOLOV6l6_yaml</li>


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
  [YOLOv6l_yaml.pt](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6l_yaml_new.pt)<br>
  [YOLOv6m_yaml.pt](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6m_yaml_new.pt)<br>
  [YOLOv6s_yaml.pt](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6s_yaml_new.pt)<br>
  [YOLOv6t_yaml.pt](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6t_yaml_new.pt)<br>
  [YOLOv6l6_p2_yaml.pt](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6l6_p2_yaml_new.pt)<br>
  [YOLOv6l6_yaml.pt](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6l6_yaml_new.pt)<br>
## 训练命令
YOLOv6t
```shell
python tools/train.py --conf-configs/model_yaml/yolov6t_yaml.py --data data/car.yaml --device 0 --img 640
```
YOLOv6s
```shell
python tools/train.py --conf-configs/model_yaml/yolov6s_yaml.py --data data/car.yaml --device 0 --img 640
```
YOLOv6m
```shell
python tools/train.py --conf-configs/model_yaml/yolov6m_yaml.py --data data/car.yaml --device 0 --img 640
```
YOLOv6l
```shell
python tools/train.py --conf-configs/model_yaml/yolov6l_yaml.py --data data/car.yaml --device 0 --img 640
```
