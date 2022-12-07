# YOLOv6n model
model = dict(
    type='YOLOv6n6',
    pretrained=None,
    build_type='yaml',
    yaml_file='configs/yaml/yolov6n6.yaml',
    depth_multiple=0.33,
    width_multiple=0.25,
    head=dict(
        type='EffiDeHead',
        num_layers=4,
        anchors=1,
        strides=[8, 16, 32, 64],
        iou_type='siou',
        use_dfl=False,
        reg_max=0 #if use_dfl is False, please set reg_max to 0
    )
)
solver = dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.02, #0.01 # 0.02
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1
)

data_aug = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
)
