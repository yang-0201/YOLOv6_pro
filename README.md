# YOLOv6 pro
[**MAF-YOLO | 我们开发的MAF-YOLO（PRCV 2024 Oral）已开源**](https://github.com/yang-0201/MAF-YOLO)
初衷让 `YOLOv6` 更换网络结构更为便捷 <br><br>
基于官方 `YOLOv6` 的整体架构，使用 `YOLOv5` 的网络构建方式构建一个 `YOLOv6` 网络，包括 `backbone`，`neck`，`effidehead` 结构 <br><br>
可以在 `yaml` 文件中任意修改或添加模块，并且每个修改的文件都是独立可运行的，目的是为了助力科研 <br><br>
后续会基于 `yolov5` 和 `yoloair` 中的模块加入更多的网络结构改进，有问题或者是改进意见都可以随时在 issues 中提出呀<br><br>
预训练权重已经从官方权重转换，确保可以匹配 <br><br>
我们使用的 `yoloair` 和 `YOLOv6 pro` 框架在 IEEE UV 2022 "Vision Meets Alage" 目标检测竞赛中取得第一名！<br><br>
博客介绍地址：<br>
[**YOLOv6 Pro | 使 YOLOv6 构建网络和更换模块更为便捷，助力科研中的网络结构改进，包括Backbone,Neck,DecoupleHead（参考YOLOv5搭建网络的方式）**](https://blog.csdn.net/qq_43000647/article/details/128212340?spm=1001.2014.3001.5502)
<p align="center">
  <img src="https://user-images.githubusercontent.com/89863442/205481197-8ad8a8e0-1f78-4ac4-a29c-efc8a39ccd10.png" align="middle" width = "1000" />
</p>

## 已经支持的模型:
<details>
<summary>基础模型</summary>
<li>YOLOV6l</li>
<li>YOLOV6m</li>
<li>YOLOV6s</li>
<li>YOLOV6t</li>
<li>YOLOV6n</li>
</details>
<details>
<summary>大尺寸模型，四个输出层</summary>
<li>YOLOV6l_p2</li>
<li>YOLOV6l6</li>
<li>YOLOV6n6</li>
</details>

YOLOv6 Pro 结构：
<li>增加 DAMO YOLO 中的 neck：RepGFPN M,S,T</li>
<li>增加 RepGhost 模块和 RepGhostNet1.0× backbone</li>
<li>增加 FocalTransformer, FocalTRC3模块</li>
<li>增加 CoAtNet 主干, MBConv, CoAtTran 模块</li>
<li>增加 SimOTA标签分配策略</li></br>

<details>
<summary>版本更新说明</summary>
<li> [ 2022/12/4 ] v1.0.0 版本，对齐完善了几个基础模型的大小和精度，构建 YOLOv6l6，YOLOv6n6，增加 wandb 记录模型训练曲线 </li>
<li> [ 2022/12/8 ]增加 GiraffeNeckV2，RepGhost，增加 VOC 数据集上的 YOLOv6 Pro 基准实验结果</li>
<li> [ 2022/12/10 ]增加 DAMO YOLO 中的 neck RepGFPN M,S,T</li>
<li> [ 2022/12/12 ]增加 FocalTransformer 中的模块，并组合为 FocalC3 模块</li>
<li> [ 2022/12/14 ]增加 RepGhost，CoAtNet 模块，其中 coatNet 包括两个 MBConv 层和两个 CoAtTran 层，组成卷积+ transformer 的主干网络结构</li>
<li> [ 2022/12/16 ]增加YOLOv6 第一个版本使用的 SimOTA 标签分配策略</li>
</details>

## Benchmark
| Model                                                        | Size | mAP<sup>val<br/>0.5:0.95              | Speed<sup>T4<br/>trt fp16 b1 <br/>(fps) | Speed<sup>T4<br/>trt fp16 b32 <br/>(fps) | Params<br/><sup> (M) | FLOPs<br/><sup> (G) |
| :----------------------------------------------------------- | ---- | :------------------------------------ | --------------------------------------- | ---------------------------------------- | -------------------- | ------------------- |
| [**YOLOv6-N**](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6n_yaml_new.pt) | 640  | 35.9<sup>300e</sup><br/>36.3<sup>400e | 802                                     | 1234                                     | 4.3                  | 11.1                |
| [**YOLOv6-T**](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6t_yaml_new.pt) | 640  | 40.3<sup>300e</sup><br/>41.1<sup>400e | 449                                     | 659                                      | 9.7                 | 24.9                |
| [**YOLOv6-S**](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6s_yaml_new.pt) | 640  | 43.5<sup>300e</sup><br/>43.8<sup>400e | 358                                     | 495                                      | 17.2                 | 44.2                |
| [**YOLOv6-M**](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6m_yaml_new.pt) | 640  | 49.5                                  | 179                                     | 233                                      | 34.3                 | 82.2                |
| [**YOLOv6-L-ReLU**](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6l_yaml_new.pt) | 640  | 51.7                                  | 113                                     | 149                                      | 58.5                 | 144.0               |
| [**YOLOv6-L**](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6l_yaml_new.pt) | 640  | 52.5                                  | 98                                      | 121                                      | 58.5                 | 144.0               |
- Speed is tested with TensorRT 7.2 on T4.
- Data from YOLOv6 official
- 目前 yolov6l，yolov6s，yolov6t，yolov6n 模型大小与精度已经和官方对齐

### YOLOv6 基准实验
数据集: VOC2007，VOC2012 </br>
训练集: train_2007, tran_2012, val_2007, test_2007 (16551 images) </br>
验证集: test_2007 (4952 images) </br>
GPU: 4090 24GB</br>
 使用 COCO 预训练权重：
 Model  | img size  | MAP0.5 | MAP0.5:0.95 | 预训练模型| epochs
 ---- | ----- | ------  | ------ | ------ | ------
 YOLOv6L  | 640 | 0.928 | 0.761 | COCO | 50 
 YOLOv6M  | 640 | 0.911 | 0.735  | COCO | 50
 YOLOv6S  | 640 | 0.902 | 0.705  | COCO | 50
 YOLOv6T  | 640 | 0.877 | 0.677  | COCO | 50
 YOLOv6N  | 640 | 0.844 | 0.633  | COCO | 50

<details>
<summary>无预训练权重</summary>

 Model  | img size  | MAP0.5 | MAP0.5:0.95 | 预训练模型| epochs
 ---- | ----- | ------  | ------ | ------ | ------
 YOLOv6N  | 640 | 0.763 | 0.537  |  None | 100
 YOLOv6T  | 640 | 0.787 | 0.560  | None | 100 
 YOLOv6S  | 640 | 0.795 | 0.567  | None | 100
 YOLOv6M  | 640 | 0.836 | 0.629  | None | 100
 YOLOv6L  | 640 | 0.862 | 0.664  | None | 100
 YOLOv6N6  | 640 | 0.799 | 0.567  | not all | 100
 YOLOv6L6  | 640 | 0.911 | 0.741  | not all | 100
 YOLOv6L6  | 640 | 0.866 | 0.674  | None | 100
 
</details>
  
### YOLOv6 Pro 改进模块的基准实验
 
 Model  | img size  | MAP0.5 | MAP0.5:0.95 | 预训练模型| epochs | Params(M)| GFLOPs | 数据集
 ---- | ----- | ------  | ------ | ------ | ------ | ------ | ------ | ------
 YOLOv6T  | 640 | 0.787 | 0.560  | None | 100 | 9.68 | 24.84 | VOC
 YOLOv6S  | 640 | 0.795 | 0.567  | None | 100 | 17.19 | 44.09 | VOC
 YOLOv6M  | 640 | 0.836 | 0.629  | None | 100 | 34.3 |82.2 | VOC
 YOLOv6T+FocalC3  | 640 | 0.780 | 0.544  |  None | 100 | 9.47 | 24.53 | VOC
- Params and GFLOPs are calculated in the validation phase，the calculation method uses the official YOLOv6 script
- Only 100 epochs of training, does not represent the final performance
## 训练你的数据集
### 数据集配置
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
   [-1, 1, BepC3, [1024, 6, "ConvWrapper"]],
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
  [[17, 1, Head_layers, [128， 16]], 
  [20, 1, Head_layers, [256， 16]], 
  [23, 1, Head_layers, [512， 16]],
  [[24, 25, 26], 1, Out, []]]

```

## 预训练权重（官方权重转化而来）
基本模型<br>
  [YOLOv6-L.pt](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6l_yaml_new.pt)<br>
  [YOLOv6-M.pt](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6m_yaml_new.pt)<br>
  [YOLOv6-S.pt](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6s_yaml_new.pt)<br>
  [YOLOv6-T.pt](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6t_yaml_new.pt)<br>
  [YOLOv6-N.pt](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6n_yaml_new.pt)<br>
大尺寸模型<br>
  [YOLOv6-L6-p2.pt](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6l6_p2_yaml_new.pt)<br>
  [YOLOv6-L6.pt](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6l6_yaml_new.pt)<br>
  [YOLOv6-N6.pt](https://github.com/yang-0201/YOLOv6_pro/releases/download/v0.0.2/yolov6n6_yaml_new.pt)<br>
tips：其中大尺寸模型无 coco 预训练权重，而是从小模型的对应层转化而来
## 训练命令
### 使用预训练权重：
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
### 不使用预训练权重：
  Tips: 如果不使用预训练权重，建议用更大的学习率，可以更快得到结果，否则会很难训练，两者配置文件的区别可以从 configs/model_yaml 和 configs/without_weights中看到</br></br>
只需要将 --conf-file configs/without_weights/yolov6l_yaml.py 中的 model_yaml 改为 without_weights 即可，调用无预训练权重的配置文件，如训练 YOLOv6l 不使用预训练：
```shell
python tools/train.py --conf-file configs/without_weights/yolov6l_yaml.py --data data/data.yaml --device 0 --img 640
```
### 如何增加自己的模块
与 yolov5 的方式类似<br><br>
step1: 先在``` yolov6/layers/common.py ``` 中加入模块的代码<br>
step2: 在``` yolov6/models/yolo.py ``` 的 parse_model 函数中加入对应模块的条件判断语句<br>
step3: 在``` configs/yaml/ ```目录下新建你的 yaml 文件，并将模块加入<br>
step4: 在``` configs/model_yaml/ ``` 目录下新建一个 py 文件，并将``` yaml_file ```目录改为 yaml 文件的路径<br>
step5: 运行训练命令<br>
### Acknowledgements
* [https://github.com/meituan/YOLOv6](https://github.com/meituan/YOLOv6)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/iscyy/yoloair](https://github.com/iscyy/yoloair)
### Cite
```shell
@article{li2022yolov6,
  title={YOLOv6: A single-stage object detection framework for industrial applications},
  author={Li, Chuyi and Li, Lulu and Jiang, Hongliang and Weng, Kaiheng and Geng, Yifei and Li, Liang and Ke, Zaidan and Li, Qingyuan and Cheng, Meng and Nie, Weiqiang and others},
  journal={arXiv preprint arXiv:2209.02976},
  year={2022}
}
```
