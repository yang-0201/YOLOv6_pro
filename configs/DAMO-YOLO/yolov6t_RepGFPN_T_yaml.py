# YOLOv6l model
model = dict(
    type='YOLOv6t_yaml',
    # pretrained='yolov6l_yaml.pt',
    pretrained="weights/yolov6t_yaml_new.pt",  #only backbone
    build_type = 'yaml',
    yaml_file = 'configs/DAMO-YOLO/yaml/yolov6t_RepGFPN_T.yaml',
    depth_multiple=0.33,
    width_multiple=0.375,
    head=dict(
        type='EffiDeHead',
        num_layers=3,
        anchors=1,
        strides=[8, 16, 32],
        iou_type='siou',
        use_dfl=False,
        reg_max=0  # if use_dfl is False, please set reg_max to 0
    )
)

solver = dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.0032,
    lrf=0.12,
    momentum=0.843,
    weight_decay=0.00036,
    warmup_epochs=2.0,
    warmup_momentum=0.5,
    warmup_bias_lr=0.05
)

data_aug = dict(
    hsv_h=0.0138,
    hsv_s=0.664,
    hsv_v=0.464,
    degrees=0.373,
    translate=0.245,
    scale=0.898,
    shear=0.602,
    flipud=0.00856,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.243,
)
