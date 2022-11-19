# YOLOv6l model
model = dict(
    type='YOLOv6l_yaml',
    pretrained='yolov6l_yaml.pt',
    build_type = 'yaml',
    yaml_file = 'configs/yaml/yolov6l.yaml',
    depth_multiple=1.0,
    width_multiple=1.0,
    head=dict(
        type='EffiDeHead',
        effidehead_channels =[64, 128, 256, 512, 1024, 256, 128, 128, 256, 256, 512],
        in_channels=[128, 256, 512],
        num_layers=3,
        begin_indices=24,
        anchors=1,
        out_indices=[17, 20, 23],
        strides=[8, 16, 32],
        iou_type='giou',
        use_dfl=True,
        reg_max=16, #if use_dfl is False, please set reg_max to 0
        distill_weight={
            'class': 2.0,
            'dfl': 1.0,
        },
    )
)

solver = dict(
    optim='Adam',
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
training_mode = "conv_silu"
# use normal conv to speed up training and further improve accuracy.
