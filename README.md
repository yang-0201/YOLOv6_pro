# YOLOv6_pro
让 `YOLOv6` 更换网络结构更为便捷 <br><br>
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
train: data/images/train # 训练集路径
val: data/images/val # 验证集路径
is_coco: False
nc: 3  # 设置为你的类别数量
names: ["car","person","bike"] #类别名称
```
### 网络结构文件配置
以yolov6l.yaml为例
```shell
depth_multiple: 1.0  # model depth multiple
width_multiple: 1.0  # layer channel multiple
backbone:
  # [from, number, module, args]
  [[-1, 1, ConvWrapper, [64, 3, 2]],  # 0-P1/2
   [-1, 1, ConvWrapper, [128, 3, 2]],  # 1-P2/4
   [-1, 1, BepC3, [128, 6, "ConvWrapper"]],
   [-1, 1, ConvWrapper, [256, 3, 2]],  # 3-P3/8
   [-1, 1, BepC3, [256, 12, "ConvWrapper"]],
   [-1, 1, ConvWrapper, [512, 3, 2]],  # 5-P4/16
   [-1, 1, BepC3, [512, 18, "ConvWrapper"]],
   [-1, 1, ConvWrapper, [1024, 3, 2]],  # 7-P5/32
   [-1, 1, BepC3, [1024,6, "ConvWrapper"]],
   [-1, 1, SPPF, [1024, 5]]]  # 9
neck:
   [[-1, 1, SimConv, [256, 1, 1]],
   [-1, 1, Transpose, [256]],
   [[-1, 6], 1, Concat, [1]],  #768
   [-1, 1, BepC3, [256, 12, "ConvWrapper"]],

   [-1, 1, SimConv, [128, 1, 1]],
   [-1, 1, Transpose, [128]],
   [[-1, 4], 1, Concat, [1]],  #384
   [-1, 1, BepC3, [128, 12, "ConvWrapper"]],   #17 (P3/8-small)

   [-1, 1, SimConv, [128, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  
   [-1, 1, BepC3, [256, 12, "ConvWrapper"]],  # 20 (P4/16-medium)

   [-1, 1, SimConv, [256, 3, 2]],
   [[-1, 10], 1, Concat, [1]], 
   [-1, 1, BepC3, [512, 12, "ConvWrapper"]]]  # 23 (P5/32-large)
effidehead:
  [[17, 1, Head_layers, [128]], 如果为m，l模型 并且使用dfl损失，该项设置为[通道数，16，你的类别数]
  [20, 1, Head_layers, [256]], 如果为s，t模型 并且不使用dfl损失，该项需要设置为[通道数，0，你的类别数]
  [23, 1, Head_layers, [512]],
  [[24, 25, 26], 1, Out, []]]

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
python tools/train.py --conf-file configs/model_yaml/yolov6t_yaml.py --data data/data.yaml --device 0 --img 640
```
YOLOv6s
```shell
python tools/train.py --conf-file configs/model_yaml/yolov6s_yaml.py --data data/data.yaml --device 0 --img 640
```
YOLOv6m
```shell
python tools/train.py --conf-file configs/model_yaml/yolov6m_yaml.py --data data/data.yaml --device 0 --img 640
```
YOLOv6l
```shell
python tools/train.py --conf-file configs/model_yaml/yolov6l_yaml.py --data data/data.yaml --device 0 --img 640
```
## Acknowledgements
* [https://github.com/meituan/YOLOv6](https://github.com/meituan/YOLOv6)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/iscyy/yoloair](https://github.com/iscyy/yoloair)
