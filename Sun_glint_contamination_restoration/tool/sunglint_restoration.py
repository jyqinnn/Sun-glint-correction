import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import argparse
import os
import cv2
import glob
import copy
import numpy as np
import torch
import imutils
import imageio
from PIL import Image
import scipy.ndimage
import torchvision.transforms.functional as F

from RAFT import utils
from RAFT import RAFT
from GMA.network import RAFTGMA

import utils.region_fill as rf
from utils.Poisson_blend_img import Poisson_blend_img
from get_flowNN_gradient import get_flowNN_gradient
from spatial_inpaint import spatial_inpaint_FMM
from spatial_inpaint import spatial_inpaint_DeepFillv2
from DeepFill_inpaint import DeepFillv2


def detectAndDescribe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if imutils.is_cv3(or_better=True):

        descriptor = cv2.xfeatures2d.SURF_create()
        (kps, features) = descriptor.detectAndCompute(image.astype(np.uint8), None)

    else:

        detector = cv2.FeatureDetector_create("SIFT")
        kps = detector.detect(gray)
        
        extractor = cv2.DescriptorExtractor_create("SIFT")
        (kps, features) = extractor.compute(gray, kps)

    kps = np.float32([kp.pt for kp in kps])

    return (kps, features)


def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio=0.75, reprojThresh=4.0):

    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []

    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    if len(matches) > 4:
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])

        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

        return (matches, H, status)

    return None


def homograpy(image1, image2):
    image1 = image1[0].permute(1, 2, 0).cpu().numpy()
    image2 = image2[0].permute(1, 2, 0).cpu().numpy()

    imgH, imgW, _ = image1.shape

    (kpsA, featuresA) = detectAndDescribe(image1)
    (kpsB, featuresB) = detectAndDescribe(image2)

    try:
        (_, H_BA, _) = matchKeypoints(kpsB, kpsA, featuresB, featuresA)
    except:
        H_BA = np.array([1.0,0,0,0,1.0,0,0,0,1.0]).reshape(3,3)

    NoneType = type(None)
    if type(H_BA) == NoneType:
        H_BA = np.array([1.0,0,0,0,1.0,0,0,0,1.0]).reshape(3,3)

    try:
        tmp = np.linalg.inv(H_BA)
    except:
        H_BA = np.array([1.0,0,0,0,1.0,0,0,0,1.0]).reshape(3,3)

    image2_registered = cv2.warpPerspective(image2, H_BA, (imgW, imgH))

    return image2_registered, H_BA


def to_tensor(img):
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    return img_t


def gradient_mask(mask):

    gradient_mask = np.logical_or.reduce((mask,
        np.concatenate((mask[1:, :], np.zeros((1, mask.shape[1]), dtype=bool)), axis=0),
        np.concatenate((mask[:, 1:], np.zeros((mask.shape[0], 1), dtype=bool)), axis=1)))

    return gradient_mask


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def initialize_model(args):
    if args.model_name == 'RAFT':
        model = torch.nn.DataParallel(RAFT(args))
    elif args.model_name == 'GMA':
        model = torch.nn.DataParallel(RAFTGMA(args))
        
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to('cuda')
    model.eval()

    return model


def infer_flow(args, mode, filename, image1, image2, imgH, imgW, model, homography=False):

    if not homography:
        _, flow = model(image1, image2, iters=20, test_mode=True)
        flow = flow[0].permute(1, 2, 0).cpu().numpy()
    else:
        image2_reg, H_BA = homograpy(image1, image2)
        image2_reg = torch.tensor(image2_reg).permute(2, 0, 1)[None].float().to('cuda')
        _, flow = model(image1, image2_reg, iters=20, test_mode=True)
        flow = flow[0].permute(1, 2, 0).cpu().numpy()

        (fy, fx) = np.mgrid[0 : imgH, 0 : imgW].astype(np.float32)

        fxx = copy.deepcopy(fx) + flow[:, :, 0]
        fyy = copy.deepcopy(fy) + flow[:, :, 1]

        (fxxx, fyyy, fz) = np.linalg.inv(H_BA).dot(np.concatenate((fxx.reshape(1, -1),
                                                   fyy.reshape(1, -1),
                                                   np.ones_like(fyy).reshape(1, -1)), axis=0))
        fxxx, fyyy = fxxx / fz, fyyy / fz

        flow = np.concatenate((fxxx.reshape(imgH, imgW, 1) - fx.reshape(imgH, imgW, 1),
                               fyyy.reshape(imgH, imgW, 1) - fy.reshape(imgH, imgW, 1)), axis=2)

    Image.fromarray(utils.flow_viz.flow_to_image(flow)).save(os.path.join(args.outroot, 'flow', mode + '_png', filename + '.png'))
    utils.frame_utils.writeFlow(os.path.join(args.outroot, 'flow', mode + '_flo', filename + '.flo'), flow)

    return flow


def calculate_flow(args, model, video):
    nFrame, _, imgH, imgW = video.shape
    FlowF = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)
    FlowB = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)

    mode_list = ['forward', 'backward']

    for mode in mode_list:
        create_dir(os.path.join(args.outroot, 'flow', mode + '_flo'))
        create_dir(os.path.join(args.outroot, 'flow', mode + '_png'))

        with torch.no_grad():
            for i in range(nFrame):
                if mode == 'forward':
                    if i == nFrame - 1:
                        continue

                    print("Calculating {0} flow {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')
                    image1 = video[i, None]
                    image2 = video[i + 1, None]
                    flow = infer_flow(args, mode, '%05d'%i, image1, image2, imgH, imgW, model, homography=False)
                    FlowF = np.concatenate((FlowF, flow[..., None]), axis=-1)
                elif mode == 'backward':
                    if i == nFrame - 1:
                        continue
                    
                    print("Calculating {0} flow {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')
                    image1 = video[i + 1, None]
                    image2 = video[i, None]
                    flow = infer_flow(args, mode, '%05d'%i, image1, image2, imgH, imgW, model, homography=False)
                    FlowB = np.concatenate((FlowB, flow[..., None]), axis=-1)

    return FlowF, FlowB


def extrapolation(args, video_ori, corrFlowF_ori, corrFlowB_ori):

    imgH, imgW, _, nFrame = video_ori.shape

    imgH_extr = int(args.H_scale * imgH)
    imgW_extr = int(args.W_scale * imgW)
    H_start = int((imgH_extr - imgH) / 2)
    W_start = int((imgW_extr - imgW) / 2)

    flow_mask = np.ones(((imgH_extr, imgW_extr)), dtype=bool)
    flow_mask[H_start : H_start + imgH, W_start : W_start + imgW] = 0

    mask_dilated = gradient_mask(flow_mask)

    video = np.zeros(((imgH_extr, imgW_extr, 3, nFrame)), dtype=np.float32)
    video[H_start : H_start + imgH, W_start : W_start + imgW, :, :] = video_ori

    for i in range(nFrame):
        print("Preparing frame {0}".format(i), '\r', end='')
        video[:, :, :, i] = cv2.inpaint((video[:, :, :, i] * 255).astype(np.uint8), flow_mask.astype(np.uint8), 3, cv2.INPAINT_TELEA).astype(np.float32)  / 255.

    corrFlowF = np.zeros(((imgH_extr, imgW_extr, 2, nFrame - 1)), dtype=np.float32)
    corrFlowB = np.zeros(((imgH_extr, imgW_extr, 2, nFrame - 1)), dtype=np.float32)

    corrFlowF[H_start : H_start + imgH, W_start : W_start + imgW] = corrFlowF_ori
    corrFlowB[H_start : H_start + imgH, W_start : W_start + imgW] = corrFlowB_ori


    return video, corrFlowF, corrFlowB, flow_mask, mask_dilated, (W_start, H_start), (W_start + imgW, H_start + imgH)


def complete_flow(args, corrFlow, flow_mask, mode):
    if mode not in ['forward', 'backward']:
        raise NotImplementedError

    sh = corrFlow.shape
    imgH = sh[0]
    imgW = sh[1]
    nFrame = sh[-1]

    create_dir(os.path.join(args.outroot, 'flow_comp', mode + '_flo'))
    create_dir(os.path.join(args.outroot, 'flow_comp', mode + '_png'))

    compFlow = np.zeros(((sh)), dtype=np.float32)

    for i in range(nFrame):
        flow = corrFlow[..., i]
        if mode == 'forward':
            flow_mask_img = flow_mask[:, :, i]
        elif mode == 'backward':
            flow_mask_img = flow_mask[:, :, i + 1]
        
        if mode == 'forward' or mode == 'backward':
            flow[:, :, 0] = rf.regionfill(flow[:, :, 0], flow_mask_img)
            flow[:, :, 1] = rf.regionfill(flow[:, :, 1], flow_mask_img)
            compFlow[:, :, :, i] = flow

    return compFlow


def video_completion(args):

    flow_model = initialize_model(args)

    filename_list = glob.glob(os.path.join(args.path, '*.png')) + \
                    glob.glob(os.path.join(args.path, '*.jpg'))

    imgH, imgW = np.array(Image.open(filename_list[0])).shape[:2]
    nFrame = len(filename_list)

    video = []
    for filename in sorted(filename_list):
        video.append(torch.from_numpy(np.array(Image.open(filename)).astype(np.uint8)[..., :3]).permute(2, 0, 1).float())

    video = torch.stack(video, dim=0)
    video = video.to('cuda')

    corrFlowF, corrFlowB = calculate_flow(args, flow_model, video)
    print('\nFinish flow prediction.')
    
    video = video.permute(2, 3, 1, 0).cpu().numpy()[:, :, ::-1, :] / 255.

    filename_list = glob.glob(os.path.join(args.path_mask, '*.png')) + \
                    glob.glob(os.path.join(args.path_mask, '*.jpg'))

    mask = []
    mask_dilated = []
    flow_mask = []
    for filename in sorted(filename_list):
        mask_img = np.array(Image.open(filename).convert('L'))
        flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=15)
        flow_mask_img = cv2.morphologyEx(flow_mask_img.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
        flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(bool)
        flow_mask.append(flow_mask_img)

        mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=5)
        mask_img = scipy.ndimage.binary_fill_holes(mask_img).astype(bool)
        mask.append(mask_img)
        mask_dilated.append(gradient_mask(mask_img))

    mask = np.stack(mask, -1).astype(bool)
    mask_dilated = np.stack(mask_dilated, -1).astype(bool)
    flow_mask = np.stack(flow_mask, -1).astype(bool) 
    
    videoFlowF = complete_flow(args, corrFlowF, flow_mask, 'forward')
    videoFlowB = complete_flow(args, corrFlowB, flow_mask, 'backward')

    gradient_x = np.empty(((imgH, imgW, 3, 0)), dtype=np.float32)
    gradient_y = np.empty(((imgH, imgW, 3, 0)), dtype=np.float32)

    for indFrame in range(nFrame):
        img = video[:, :, :, indFrame]
        img[mask[:, :, indFrame], :] = 0
        img = cv2.inpaint((img * 255).astype(np.uint8), mask[:, :, indFrame].astype(np.uint8), 3, cv2.INPAINT_TELEA).astype(np.float32)  / 255.

        gradient_x_ = np.concatenate((np.diff(img, axis=1), np.zeros((imgH, 1, 3), dtype=np.float32)), axis=1)
        gradient_y_ = np.concatenate((np.diff(img, axis=0), np.zeros((1, imgW, 3), dtype=np.float32)), axis=0)
        gradient_x = np.concatenate((gradient_x, gradient_x_.reshape(imgH, imgW, 3, 1)), axis=-1)
        gradient_y = np.concatenate((gradient_y, gradient_y_.reshape(imgH, imgW, 3, 1)), axis=-1)

        gradient_x[mask_dilated[:, :, indFrame], :, indFrame] = 0
        gradient_y[mask_dilated[:, :, indFrame], :, indFrame] = 0


    iter = 0
    gradient_x_filled = gradient_x
    gradient_y_filled = gradient_y
    mask_gradient = mask_dilated
    video_comp = video

    while(np.sum(mask) > 0):
        create_dir(os.path.join(args.outroot, 'removal_' + str(iter)))

        gradient_x_filled, gradient_y_filled, mask_gradient = \
            get_flowNN_gradient(args,
                                gradient_x_filled,
                                gradient_y_filled,
                                mask,
                                mask_gradient,
                                videoFlowF,
                                videoFlowB,
                                )

        for indFrame in range(nFrame):
            mask_gradient[:, :, indFrame] = scipy.ndimage.binary_fill_holes(mask_gradient[:, :, indFrame]).astype(bool)
            
        for indFrame in range(nFrame):
            if mask[:, :, indFrame].sum() > 0:
                try:
                    frameBlend, UnfilledMask = Poisson_blend_img(video_comp[:, :, :, indFrame], gradient_x_filled[:, 0 : imgW - 1, :, indFrame], gradient_y_filled[0 : imgH - 1, :, :, indFrame], mask[:, :, indFrame], mask_gradient[:, :, indFrame])
                except:
                    frameBlend, UnfilledMask = video_comp[:, :, :, indFrame], mask[:, :, indFrame]

                frameBlend = np.clip(frameBlend, 0, 1.0)
                tmp = cv2.inpaint((frameBlend * 255).astype(np.uint8), UnfilledMask.astype(np.uint8), 3, cv2.INPAINT_TELEA).astype(np.float32) / 255.
                frameBlend[UnfilledMask, :] = tmp[UnfilledMask, :]

                video_comp[:, :, :, indFrame] = frameBlend
                mask[:, :, indFrame] = UnfilledMask

                frameBlend_ = copy.deepcopy(frameBlend)
                frameBlend_[mask[:, :, indFrame], :] = [0, 1., 0]
            else:
                frameBlend_ = video_comp[:, :, :, indFrame]

            cv2.imwrite(os.path.join(args.outroot, 'removal_' + str(iter), '%05d.png'%indFrame), frameBlend_ * 255.)
            
            
        if args.inpaint_model == 'FMM':
            mask, video_comp = spatial_inpaint_FMM(mask, video_comp)
            
        elif args.inpaint_model == 'deepfill':
            deepfill = DeepFillv2(pretrained_model=args.deepfill_model, image_shape=[imgH, imgW])
            mask, video_comp = spatial_inpaint_DeepFillv2(deepfill, mask, video_comp)
    
        iter += 1

        for indFrame in range(nFrame):
            mask_gradient[:, :, indFrame] = gradient_mask(mask[:, :, indFrame])

            gradient_x_filled[:, :, :, indFrame] = np.concatenate((np.diff(video_comp[:, :, :, indFrame], axis=1), np.zeros((imgH, 1, 3), dtype=np.float32)), axis=1)
            gradient_y_filled[:, :, :, indFrame] = np.concatenate((np.diff(video_comp[:, :, :, indFrame], axis=0), np.zeros((1, imgW, 3), dtype=np.float32)), axis=0)

            gradient_x_filled[mask_gradient[:, :, indFrame], :, indFrame] = 0
            gradient_y_filled[mask_gradient[:, :, indFrame], :, indFrame] = 0

    create_dir(os.path.join(args.outroot, 'removal_' + 'final'))
    for i in range(nFrame):
        img = video_comp[:, :, :, i] * 255
        cv2.imwrite(os.path.join(args.outroot, 'removal_' + 'final', '%05d.png'%i), img)


def main(args):

    video_completion(args)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # video completion
    parser.add_argument('--path', default='./data/cropglint', help="dataset for evaluation")
    parser.add_argument('--path_mask', default='./data/croplabel', help="mask for object removal")
    parser.add_argument('--outroot', default='./result/cropglint_removal', help="output directory")
    parser.add_argument('--consistencyThres', dest='consistencyThres', default=np.inf, type=float, help='flow consistency error threshold')
    parser.add_argument('--alpha', dest='alpha', default=0.1, type=float)
 
    # RAFT/GMA
    parser.add_argument('--model', default='../weight/gma-things.pth', help="restore checkpoint")
    parser.add_argument('--model_name', default='GMA', choices=['GMA', 'RAFT'], help= "optical model")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--num_heads', default=1, type=int, help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true', help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true', help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # FMM/Deepfill
    parser.add_argument('--inpaint_model', default='FMM', choices=['deepfill', 'FMM'], help="inpaint model")
    parser.add_argument('--deepfill_model', default='./weight/imagenet_deepfill.pth', help="restore checkpoint")

    # extrapolation
    parser.add_argument('--H_scale', dest='H_scale', default=2, type=float, help='H extrapolation scale')
    parser.add_argument('--W_scale', dest='W_scale', default=2, type=float, help='W extrapolation scale')

    args = parser.parse_args()

    main(args)
