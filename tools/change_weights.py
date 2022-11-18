from collections import OrderedDict
import torch
ckpt = torch.load("weights/yolov6l.pt")

def change_swin():
    state_dict = OrderedDict()
    list = [1,2,3,4,5,6,7,8,9,10]
    for i in list:
        for key, weight in ckpt['state_dict'].items():
        # for key, weight in ckpt["model"].items():
            print(key)
        #########
            new_key_origin = key.split(".")[1:]
            new_key_origin ='.'.join(new_key_origin)
        ###########
            new_key = "model."+str(i)+"."+new_key_origin
            state_dict[new_key] = weight
            print(f'Convert {key} to {new_key}')

    checkpoint = dict()
    checkpoint['model'] = state_dict
    torch.save(checkpoint, "yolov5l6_cbv2_swin_small.pt")
def change_yolov5():
    state_dict = OrderedDict()

    for key, weight in ckpt['model'].state_dict().items():
    # for key, weight in ckpt["model"].items():
        print(key)
    #########
        new_key_origin = key.split(".")[1:]
        new_key_origin ='.'.join(new_key_origin)
    ###########
        new_key = "backbone."+new_key_origin
        state_dict[new_key] = weight
        print(f'Convert {key} to {new_key}')

    checkpoint = dict()
    checkpoint['model'] = state_dict
    torch.save(checkpoint, "yolov5l_backbone.pt")
def change_yolov6l():
    state_dict = OrderedDict()
    stem = []
    new_key_more = []
    ckpt_state = ckpt['model'].state_dict()
    for key, weight in ckpt['model'].state_dict().items():
        new_key = ""
    #########
        new_key_origin = key.split(".")[0:]
        if new_key_origin[1] == "stem":
            new_key_origin[1] = "0"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "ERBlock_2":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "1"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "2"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_3":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "3"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "4"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_4":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "5"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "6"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_5":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "7"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "8"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "9"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "Rep_p4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "13"
            new_key = '.'.join(new_key_origin)

        elif new_key_origin[1] == "Rep_p3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "17"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "20"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "23"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "10"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "11"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "14"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "15"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample2":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "18"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "21"
            new_key = '.'.join(new_key_origin)
        else:
            new_key = key


    ###########

        state_dict[new_key] = weight
        print(f'Convert {key} to {new_key}')

    checkpoint = dict()
    checkpoint['model'] = state_dict
    torch.save(checkpoint, "yolov6l_yaml.pt")
change_yolov6l()