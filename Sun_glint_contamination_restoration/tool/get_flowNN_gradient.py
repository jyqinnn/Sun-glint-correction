from __future__ import absolute_import, division, print_function, unicode_literals
import os
import cv2
import copy
import numpy as np
import scipy.io as sio
from utils.common_utils import interp, BFconsistCheck, \
    FBconsistCheck, consistCheck, get_KeySourceFrame_flowNN_gradient


def get_flowNN_gradient(args,
                        gradient_x,
                        gradient_y,
                        mask_RGB,
                        mask,
                        videoFlowF,
                        videoFlowB,
                        ):


    
    num_candidate = 2
    imgH, imgW, nFrame = mask.shape
    numPix = np.sum(mask)

    sub = np.concatenate((np.where(mask == 1)[0].reshape(-1, 1),
                          np.where(mask == 1)[1].reshape(-1, 1),
                          np.where(mask == 1)[2].reshape(-1, 1)), axis=1)

    flowNN = np.ones((numPix, 3, 2)) * 99999   # * -1
    HaveFlowNN = np.ones((imgH, imgW, nFrame, 2)) * 99999
    HaveFlowNN[mask, :] = 0
    numPixInd = np.ones((imgH, imgW, nFrame)) * -1
    consistencyMap = np.zeros((imgH, imgW, num_candidate, nFrame))
    consistency_uv = np.zeros((imgH, imgW, 2, 2, nFrame))

    for idx in range(len(sub)):
        numPixInd[sub[idx, 0], sub[idx, 1], sub[idx, 2]] = idx

    frameIndSetF = range(1, nFrame)
    frameIndSetB = range(nFrame - 2, -1, -1)
    
    print('Forward Pass......')

    NN_idx = 0
    for indFrame in frameIndSetF:
        holepixPosInd = (sub[:, 2] == indFrame)
        holepixPos = sub[holepixPosInd, :]
        flowB_neighbor = copy.deepcopy(holepixPos)
        flowB_neighbor = flowB_neighbor.astype(np.float32)

        flowB_vertical = videoFlowB[:, :, 1, indFrame - 1]
        flowB_horizont = videoFlowB[:, :, 0, indFrame - 1]
        flowF_vertical = videoFlowF[:, :, 1, indFrame - 1]
        flowF_horizont = videoFlowF[:, :, 0, indFrame - 1]

        flowB_neighbor[:, 0] += flowB_vertical[holepixPos[:, 0], holepixPos[:, 1]]
        flowB_neighbor[:, 1] += flowB_horizont[holepixPos[:, 0], holepixPos[:, 1]]
        flowB_neighbor[:, 2] -= 1

        flow_neighbor_int = np.round(copy.deepcopy(flowB_neighbor)).astype(np.int32)

        IsConsist, _ = BFconsistCheck(flowB_neighbor,
                                      flowF_vertical,
                                      flowF_horizont,
                                      holepixPos,
                                      args.consistencyThres)

        BFdiff, BF_uv = consistCheck(videoFlowF[:, :, :, indFrame - 1],
                                     videoFlowB[:, :, :, indFrame - 1])

        ValidPos = np.logical_and(
            np.logical_and(flow_neighbor_int[:, 0] >= 0,
                           flow_neighbor_int[:, 0] < imgH - 1),
            np.logical_and(flow_neighbor_int[:, 1] >= 0,
                           flow_neighbor_int[:, 1] < imgW - 1))

        holepixPos = holepixPos[ValidPos, :]
        flowB_neighbor = flowB_neighbor[ValidPos, :]
        flow_neighbor_int = flow_neighbor_int[ValidPos, :]
        IsConsist = IsConsist[ValidPos]

        KnownInd = mask[flow_neighbor_int[:, 0],
                        flow_neighbor_int[:, 1],
                        indFrame - 1] == 0

        KnownIsConsist = np.logical_and(KnownInd, IsConsist)

        flowNN[numPixInd[holepixPos[KnownIsConsist, 0],
                         holepixPos[KnownIsConsist, 1],
                         indFrame].astype(np.int32), :, NN_idx] = \
                                                flowB_neighbor[KnownIsConsist, :]
        HaveFlowNN[holepixPos[KnownIsConsist, 0],
                   holepixPos[KnownIsConsist, 1],
                   indFrame,
                   NN_idx] = 1

        consistency_uv[holepixPos[KnownIsConsist, 0], holepixPos[KnownIsConsist, 1], NN_idx, 0, indFrame] = np.abs(BF_uv[holepixPos[KnownIsConsist, 0], holepixPos[KnownIsConsist, 1], 0])
        consistency_uv[holepixPos[KnownIsConsist, 0], holepixPos[KnownIsConsist, 1], NN_idx, 1, indFrame] = np.abs(BF_uv[holepixPos[KnownIsConsist, 0], holepixPos[KnownIsConsist, 1], 1])

        UnknownInd = np.invert(KnownInd)

        HaveNNInd = HaveFlowNN[flow_neighbor_int[:, 0],
                               flow_neighbor_int[:, 1],
                               indFrame - 1,
                               NN_idx] == 1

        Valid_ = np.logical_and.reduce((UnknownInd, HaveNNInd, IsConsist))

        refineVec = np.concatenate((
            (flowB_neighbor[:, 0] - flow_neighbor_int[:, 0]).reshape(-1, 1),
            (flowB_neighbor[:, 1] - flow_neighbor_int[:, 1]).reshape(-1, 1),
            np.zeros((flowB_neighbor[:, 0].shape[0])).reshape(-1, 1)), 1)

        flowNN_tmp = copy.deepcopy(flowNN[numPixInd[flow_neighbor_int[:, 0],
                                                    flow_neighbor_int[:, 1],
                                                    indFrame - 1].astype(np.int32), :, NN_idx] + refineVec[:, :])
        flowNN_tmp = np.round(flowNN_tmp).astype(np.int32)

        ValidPos_ = np.logical_and(
            np.logical_and(flowNN_tmp[:, 0] >= 0,
                           flowNN_tmp[:, 0] < imgH - 1),
            np.logical_and(flowNN_tmp[:, 1] >= 0,
                           flowNN_tmp[:, 1] < imgW - 1))

        flowNN_tmp[np.invert(ValidPos_), :] = 0
        ValidNN = mask[flowNN_tmp[:, 0],
                       flowNN_tmp[:, 1],
                       flowNN_tmp[:, 2]] == 0

        Valid = np.logical_and.reduce((Valid_, ValidPos_))

        flowNN[numPixInd[holepixPos[Valid, 0],
                         holepixPos[Valid, 1],
                         indFrame].astype(np.int32), :, NN_idx] = \
        flowNN[numPixInd[flow_neighbor_int[Valid, 0],
                         flow_neighbor_int[Valid, 1],
                         indFrame - 1].astype(np.int32), :, NN_idx] + refineVec[Valid, :]

        HaveFlowNN[holepixPos[Valid, 0],
                   holepixPos[Valid, 1],
                   indFrame,
                   NN_idx] = 1

        consistency_uv[holepixPos[Valid, 0], holepixPos[Valid, 1], NN_idx, 0, indFrame] = np.maximum(np.abs(BF_uv[holepixPos[Valid, 0], holepixPos[Valid, 1], 0]), np.abs(consistency_uv[flow_neighbor_int[Valid, 0], flow_neighbor_int[Valid, 1], NN_idx, 0, indFrame - 1]))
        consistency_uv[holepixPos[Valid, 0], holepixPos[Valid, 1], NN_idx, 1, indFrame] = np.maximum(np.abs(BF_uv[holepixPos[Valid, 0], holepixPos[Valid, 1], 1]), np.abs(consistency_uv[flow_neighbor_int[Valid, 0], flow_neighbor_int[Valid, 1], NN_idx, 1, indFrame - 1]))

        consistencyMap[:, :, NN_idx, indFrame] = (consistency_uv[:, :, NN_idx, 0, indFrame] ** 2 + consistency_uv[:, :, NN_idx, 1, indFrame] ** 2) ** 0.5

    print('Backward Pass......')

    NN_idx = 1
    for indFrame in frameIndSetB:

        holepixPosInd = (sub[:, 2] == indFrame)

        holepixPos = sub[holepixPosInd, :]

        flowF_neighbor = copy.deepcopy(holepixPos)
        flowF_neighbor = flowF_neighbor.astype(np.float32)

        flowF_vertical = videoFlowF[:, :, 1, indFrame]
        flowF_horizont = videoFlowF[:, :, 0, indFrame]
        flowB_vertical = videoFlowB[:, :, 1, indFrame]
        flowB_horizont = videoFlowB[:, :, 0, indFrame]

        flowF_neighbor[:, 0] += flowF_vertical[holepixPos[:, 0], holepixPos[:, 1]]
        flowF_neighbor[:, 1] += flowF_horizont[holepixPos[:, 0], holepixPos[:, 1]]
        flowF_neighbor[:, 2] += 1

        flow_neighbor_int = np.round(copy.deepcopy(flowF_neighbor)).astype(np.int32)

        IsConsist, _ = FBconsistCheck(flowF_neighbor,
                                      flowB_vertical,
                                      flowB_horizont,
                                      holepixPos,
                                      args.consistencyThres)

        FBdiff, FB_uv = consistCheck(videoFlowB[:, :, :, indFrame],
                                     videoFlowF[:, :, :, indFrame])

        ValidPos = np.logical_and(
            np.logical_and(flow_neighbor_int[:, 0] >= 0,
                           flow_neighbor_int[:, 0] < imgH - 1),
            np.logical_and(flow_neighbor_int[:, 1] >= 0,
                           flow_neighbor_int[:, 1] < imgW - 1))

        holepixPos = holepixPos[ValidPos, :]
        flowF_neighbor = flowF_neighbor[ValidPos, :]
        flow_neighbor_int = flow_neighbor_int[ValidPos, :]
        IsConsist = IsConsist[ValidPos]

        KnownInd = mask[flow_neighbor_int[:, 0],
                        flow_neighbor_int[:, 1],
                        indFrame + 1] == 0

        KnownIsConsist = np.logical_and(KnownInd, IsConsist)
        flowNN[numPixInd[holepixPos[KnownIsConsist, 0],
                         holepixPos[KnownIsConsist, 1],
                         indFrame].astype(np.int32), :, NN_idx] = \
                                                flowF_neighbor[KnownIsConsist, :]

        HaveFlowNN[holepixPos[KnownIsConsist, 0],
                   holepixPos[KnownIsConsist, 1],
                   indFrame,
                   NN_idx] = 1

        consistency_uv[holepixPos[KnownIsConsist, 0], holepixPos[KnownIsConsist, 1], NN_idx, 0, indFrame] = np.abs(FB_uv[holepixPos[KnownIsConsist, 0], holepixPos[KnownIsConsist, 1], 0])
        consistency_uv[holepixPos[KnownIsConsist, 0], holepixPos[KnownIsConsist, 1], NN_idx, 1, indFrame] = np.abs(FB_uv[holepixPos[KnownIsConsist, 0], holepixPos[KnownIsConsist, 1], 1])

        UnknownInd = np.invert(KnownInd)
        HaveNNInd = HaveFlowNN[flow_neighbor_int[:, 0],
                               flow_neighbor_int[:, 1],
                               indFrame + 1,
                               NN_idx] == 1

        Valid_ = np.logical_and.reduce((UnknownInd, HaveNNInd, IsConsist))

        refineVec = np.concatenate((
            (flowF_neighbor[:, 0] - flow_neighbor_int[:, 0]).reshape(-1, 1),
            (flowF_neighbor[:, 1] - flow_neighbor_int[:, 1]).reshape(-1, 1),
            np.zeros((flowF_neighbor[:, 0].shape[0])).reshape(-1, 1)), 1)

        flowNN_tmp = copy.deepcopy(flowNN[numPixInd[flow_neighbor_int[:, 0],
                                                    flow_neighbor_int[:, 1],
                                                    indFrame + 1].astype(np.int32), :, NN_idx] + refineVec[:, :])
        flowNN_tmp = np.round(flowNN_tmp).astype(np.int32)

        ValidPos_ = np.logical_and(
            np.logical_and(flowNN_tmp[:, 0] >= 0,
                           flowNN_tmp[:, 0] < imgH - 1),
            np.logical_and(flowNN_tmp[:, 1] >= 0,
                           flowNN_tmp[:, 1] < imgW - 1))

        flowNN_tmp[np.invert(ValidPos_), :] = 0
        ValidNN = mask[flowNN_tmp[:, 0],
                       flowNN_tmp[:, 1],
                       flowNN_tmp[:, 2]] == 0

        Valid = np.logical_and.reduce((Valid_, ValidPos_))

        flowNN[numPixInd[holepixPos[Valid, 0],
                         holepixPos[Valid, 1],
                         indFrame].astype(np.int32), :, NN_idx] = \
        flowNN[numPixInd[flow_neighbor_int[Valid, 0],
                         flow_neighbor_int[Valid, 1],
                         indFrame + 1].astype(np.int32), :, NN_idx] + refineVec[Valid, :]

        HaveFlowNN[holepixPos[Valid, 0],
                   holepixPos[Valid, 1],
                   indFrame,
                   NN_idx] = 1

        consistency_uv[holepixPos[Valid, 0], holepixPos[Valid, 1], NN_idx, 0, indFrame] = np.maximum(np.abs(FB_uv[holepixPos[Valid, 0], holepixPos[Valid, 1], 0]), np.abs(consistency_uv[flow_neighbor_int[Valid, 0], flow_neighbor_int[Valid, 1], NN_idx, 0, indFrame + 1]))
        consistency_uv[holepixPos[Valid, 0], holepixPos[Valid, 1], NN_idx, 1, indFrame] = np.maximum(np.abs(FB_uv[holepixPos[Valid, 0], holepixPos[Valid, 1], 1]), np.abs(consistency_uv[flow_neighbor_int[Valid, 0], flow_neighbor_int[Valid, 1], NN_idx, 1, indFrame + 1]))

        consistencyMap[:, :, NN_idx, indFrame] = (consistency_uv[:, :, NN_idx, 0, indFrame] ** 2 + consistency_uv[:, :, NN_idx, 1, indFrame] ** 2) ** 0.5

    gradient_x_BN = copy.deepcopy(gradient_x)
    gradient_y_BN = copy.deepcopy(gradient_y)
    gradient_x_FN = copy.deepcopy(gradient_x)
    gradient_y_FN = copy.deepcopy(gradient_y)

    for indFrame in range(nFrame):
        SourceFmInd = np.where(flowNN[:, 2, 0] == indFrame)

        if len(SourceFmInd[0]) != 0:

            gradient_x_BN[sub[SourceFmInd[0], :][:, 0],
                    sub[SourceFmInd[0], :][:, 1],
                 :, sub[SourceFmInd[0], :][:, 2]] = \
                interp(gradient_x_BN[:, :, :, indFrame],
                        flowNN[SourceFmInd, 1, 0].reshape(-1),
                        flowNN[SourceFmInd, 0, 0].reshape(-1))

            gradient_y_BN[sub[SourceFmInd[0], :][:, 0],
                    sub[SourceFmInd[0], :][:, 1],
                 :, sub[SourceFmInd[0], :][:, 2]] = \
                interp(gradient_y_BN[:, :, :, indFrame],
                        flowNN[SourceFmInd, 1, 0].reshape(-1),
                        flowNN[SourceFmInd, 0, 0].reshape(-1))

            assert(((sub[SourceFmInd[0], :][:, 2] - indFrame) <= 0).sum() == 0)

    for indFrame in range(nFrame - 1, -1, -1):
        SourceFmInd = np.where(flowNN[:, 2, 1] == indFrame)
        if len(SourceFmInd[0]) != 0:

            gradient_x_FN[sub[SourceFmInd[0], :][:, 0],
                          sub[SourceFmInd[0], :][:, 1],
                       :, sub[SourceFmInd[0], :][:, 2]] = \
                interp(gradient_x_FN[:, :, :, indFrame],
                         flowNN[SourceFmInd, 1, 1].reshape(-1),
                         flowNN[SourceFmInd, 0, 1].reshape(-1))

            gradient_y_FN[sub[SourceFmInd[0], :][:, 0],
                    sub[SourceFmInd[0], :][:, 1],
                 :, sub[SourceFmInd[0], :][:, 2]] = \
                interp(gradient_y_FN[:, :, :, indFrame],
                         flowNN[SourceFmInd, 1, 1].reshape(-1),
                         flowNN[SourceFmInd, 0, 1].reshape(-1))

            assert(((indFrame - sub[SourceFmInd[0], :][:, 2]) <= 0).sum() == 0)

    mask_tofill = np.zeros((imgH, imgW, nFrame)).astype(np.bool)

    for indFrame in range(nFrame):

        HaveNN = np.zeros((imgH, imgW, num_candidate))

        HaveNN[:, :, 0] = HaveFlowNN[:, :, indFrame, 0] == 1
        HaveNN[:, :, 1] = HaveFlowNN[:, :, indFrame, 1] == 1

        NotHaveNN = np.logical_and(np.invert(HaveNN.astype(np.bool)),
                np.repeat(np.expand_dims((mask[:, :, indFrame]), 2), num_candidate, axis=2))

        
        HaveNN_sum = np.logical_or.reduce((HaveNN[:, :, 0],
                                            HaveNN[:, :, 1]))

        gradient_x_Candidate = np.zeros((imgH, imgW, 3, num_candidate))
        gradient_y_Candidate = np.zeros((imgH, imgW, 3, num_candidate))

        gradient_x_Candidate[:, :, :, 0] = gradient_x_BN[:, :, :, indFrame]
        gradient_y_Candidate[:, :, :, 0] = gradient_y_BN[:, :, :, indFrame]
        gradient_x_Candidate[:, :, :, 1] = gradient_x_FN[:, :, :, indFrame]
        gradient_y_Candidate[:, :, :, 1] = gradient_y_FN[:, :, :, indFrame]

        consistencyMap[:, :, :, indFrame] = np.exp( - consistencyMap[:, :, :, indFrame] / args.alpha)

        consistencyMap[NotHaveNN[:, :, 0], 0, indFrame] = 0
        consistencyMap[NotHaveNN[:, :, 1], 1, indFrame] = 0

        weights = (consistencyMap[HaveNN_sum, :, indFrame] * HaveNN[HaveNN_sum, :]) / ((consistencyMap[HaveNN_sum, :, indFrame] * HaveNN[HaveNN_sum, :]).sum(axis=1, keepdims=True))

        fix = np.where((consistencyMap[HaveNN_sum, :, indFrame] * HaveNN[HaveNN_sum, :]).sum(axis=1, keepdims=True) == 0)[0]
        weights[fix, :] = HaveNN[HaveNN_sum, :][fix, :] / HaveNN[HaveNN_sum, :][fix, :].sum(axis=1, keepdims=True)

        gradient_x[HaveNN_sum, 0, indFrame] = \
            np.sum(np.multiply(gradient_x_Candidate[HaveNN_sum, 0, :], weights), axis=1)
        gradient_x[HaveNN_sum, 1, indFrame] = \
            np.sum(np.multiply(gradient_x_Candidate[HaveNN_sum, 1, :], weights), axis=1)
        gradient_x[HaveNN_sum, 2, indFrame] = \
            np.sum(np.multiply(gradient_x_Candidate[HaveNN_sum, 2, :], weights), axis=1)

        gradient_y[HaveNN_sum, 0, indFrame] = \
            np.sum(np.multiply(gradient_y_Candidate[HaveNN_sum, 0, :], weights), axis=1)
        gradient_y[HaveNN_sum, 1, indFrame] = \
            np.sum(np.multiply(gradient_y_Candidate[HaveNN_sum, 1, :], weights), axis=1)
        gradient_y[HaveNN_sum, 2, indFrame] = \
            np.sum(np.multiply(gradient_y_Candidate[HaveNN_sum, 2, :], weights), axis=1)

        mask_tofill[np.logical_and(np.invert(HaveNN_sum), mask[:, :, indFrame]), indFrame] = True

    return gradient_x, gradient_y, mask_tofill
