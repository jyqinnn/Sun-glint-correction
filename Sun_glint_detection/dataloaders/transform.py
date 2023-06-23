import cv2
import numpy as np
import random

def rand_flip(data,label):
    m=random.randint(0,1)
    angle=random.randint(-1,1)
    if m==0:
       data=cv2.flip(data,angle)
       label=cv2.flip(label,angle)
    return data,label

def rand_trans(data,label):
    pixelx=random.randint(0,50)
    pixely = random.randint(0, 50)
    affine_arr = np.float32([[1, 0, pixelx], [0, 1, pixely]])
    data=cv2.warpAffine(data,affine_arr,(data.shape[0],data.shape[1]))
    label=cv2.warpAffine(label, affine_arr, (label.shape[0], label.shape[1]))
    return data,label

def rand_rotate(data,label):
    degree=random.randint(0,10)
    M = cv2.getRotationMatrix2D((data.shape[0] / 2, data.shape[1] / 2), degree, 1)
    data = cv2.warpAffine(data, M, (data.shape[0], data.shape[1]))
    label= cv2.warpAffine(label, M, (label.shape[0], label.shape[1]))
    return data, label

def rand_crop(data,label,H_want,W_want):
    H,W,C=data.shape
    len_H=random.randint(0,H-H_want)
    len_W=random.randint(0,W-W_want)
    data=data[len_H:(len_H+H_want),len_W:(len_W+W_want)]
    label=label[len_H:(len_H+H_want),len_W:(len_W+W_want)]
    return data, label

def rand_contrast(data): 
    decimal = random.randint(8, 12)
    c=decimal*0.1
    b=random.randint(-30, 30)
    rows, cols, chunnel = data.shape
    blank = np.zeros([rows, cols, chunnel], data.dtype)  # np.zeros(img1.shape, dtype=uint8)
    dst = cv2.addWeighted(data, c, blank, 1-c, b)
    return dst

def rand_clahe(data):
    clipLimit=random.randint(0,20)
    if(clipLimit==0):
        return data
    else:
        ycrcb = cv2.cvtColor(data, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb)
        channels=np.array(channels)
        clahe=cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
        # clipLimit
        #         tileGridSize
        channels[0]=clahe.apply(channels[0])
        cv2.merge(channels,ycrcb)
        img=cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR)
        return img

def rand_hue(data):
    hsv=cv2.cvtColor(data, cv2.COLOR_BGR2HSV)
    channels=cv2.split(hsv)
    hue=random.randint(0,10)
    hue = np.array(hue, dtype='uint8')
    h=channels[0]
    h+=hue
    h[h<0]=0
    h[h>255]=255
    cv2.merge(channels, hsv)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def rand_rotate90(img,label,prob=1.0):
    if random.random() < prob:
        factor = random.randint(0, 4)
        img = np.rot90(img, factor)
        if label is not None:
            label = np.rot90(label, factor)
        return img.copy(), label.copy()