from models.fassnet import FASSNet
import torch
import math

nclass=2

def build_model(args):
    if args.model=='fassnet':
        model=FASSNet(img_ch=3,num_classes=nclass,sync_bn=args.sync_bn)
    
    return model