# YOLOv6_pro
<p>
Make it easier for yolov6 to change the network structure.
</p>
<p>
Build a YOLOv6 network using the YOLOv5 style of building networks, modifying your modules in the yaml file as you wish.
</p>
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

<summary> 预训练权重 </summary>
[文本](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6l_yaml.pt)
<p>
  YOLOv6m_yaml.pt 
 [melt](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6l_yaml.pt)
 链接：https://pan.baidu.com/s/1DKEi84e6XfehUjA8--dLrQ?pwd=uuhn  提取码：uuhn 
</p>
<p>
  YOLOv6s_yaml.pt 链接：https://pan.baidu.com/s/1p-y6QhCslwKT9hCO0o8qEg?pwd=eott  提取码：eott 
</p>
<p>
  YOLOv6t_yaml.pt 链接：https://pan.baidu.com/s/1iWIXJvc2C6ZadekFNK0uFQ?pwd=hc1c  提取码：hc1c 
</p>
<p>
  YOLOv6l_yaml.pt 链接：https://pan.baidu.com/s/17QXSAWbGZZ7hNCJcEq2-Rw?pwd=f2nf  提取码：f2nf 
</p>
<p>
  预训练权重放在 weights 目录
</p>

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
