from __future__ import absolute_import, division, print_function, unicode_literals

import scipy.ndimage
from scipy.sparse.linalg import spsolve
from scipy import sparse
import scipy.io as sio
import numpy as np
from PIL import Image
import copy
import cv2
import os
import argparse


def sub2ind(pi, pj, imgH, imgW):
    return pj + pi * imgW


def Poisson_blend_img(imgTrg, imgSrc_gx, imgSrc_gy, holeMask, gradientMask=None):

    imgH, imgW, nCh = imgTrg.shape

    if not isinstance(gradientMask, np.ndarray):
        gradientMask = np.zeros((imgH, imgW), dtype=np.float32)

    imgRecon = np.zeros((imgH, imgW, nCh), dtype=np.float32)

    A, b, UnfilledMask = solvePoisson(holeMask, imgSrc_gx, imgSrc_gy, imgTrg,
                                                  gradientMask)
    for ch in range(nCh):

        x = scipy.sparse.linalg.lsqr(A, b[:, ch])[0]

        imgRecon[:, :, ch] = x.reshape(imgH, imgW)

    holeMaskC = np.tile(np.expand_dims(holeMask, axis=2), (1, 1, nCh))
    imgBlend = holeMaskC * imgRecon + (1 - holeMaskC) * imgTrg
    return imgBlend, UnfilledMask

def solvePoisson(holeMask, imgSrc_gx, imgSrc_gy, imgTrg,
                           gradientMask):

    UnfilledMask_topleft = copy.deepcopy(holeMask)
    UnfilledMask_bottomright = copy.deepcopy(holeMask)

    imgH, imgW = holeMask.shape
    N = imgH * imgW

    numUnknownPix = holeMask.sum()

    dx = [1, 0, -1,  0]
    dy = [0, 1,  0, -1]

    I = np.empty((0, 1), dtype=np.float32)
    J = np.empty((0, 1), dtype=np.float32)
    S = np.empty((0, 1), dtype=np.float32)

    b = np.empty((0, 3), dtype=np.float32)

    pi = np.expand_dims(np.where(holeMask == 1)[0], axis=1)
    pj = np.expand_dims(np.where(holeMask == 1)[1], axis=1)
    pind = sub2ind(pi, pj, imgH, imgW)

    qi = np.concatenate((pi + dy[0],
                         pi + dy[1],
                         pi + dy[2],
                         pi + dy[3]), axis=1)

    qj = np.concatenate((pj + dx[0],
                         pj + dx[1],
                         pj + dx[2],
                         pj + dx[3]), axis=1)

    validN = (qi >= 0) & (qi <= imgH - 1) & (qj >= 0) & (qj <= imgW - 1)
    qind = np.zeros((validN.shape), dtype=np.float32)
    qind[validN] = sub2ind(qi[validN], qj[validN], imgH, imgW)

    e_start = 0
    e_stop  = 0

    I, J, S, b, e_start, e_stop = constructEquation(0, validN, holeMask, gradientMask, imgSrc_gx, imgSrc_gy, imgTrg, pi, pj, pind, qi, qj, qind, I, J, S, b, e_start, e_stop)
    I, J, S, b, e_start, e_stop = constructEquation(1, validN, holeMask, gradientMask, imgSrc_gx, imgSrc_gy, imgTrg, pi, pj, pind, qi, qj, qind, I, J, S, b, e_start, e_stop)
    I, J, S, b, e_start, e_stop = constructEquation(2, validN, holeMask, gradientMask, imgSrc_gx, imgSrc_gy, imgTrg, pi, pj, pind, qi, qj, qind, I, J, S, b, e_start, e_stop)
    I, J, S, b, e_start, e_stop = constructEquation(3, validN, holeMask, gradientMask, imgSrc_gx, imgSrc_gy, imgTrg, pi, pj, pind, qi, qj, qind, I, J, S, b, e_start, e_stop)

    nEqn = len(b)
    A = sparse.csr_matrix((S[:, 0], (I[:, 0], J[:, 0])), shape=(nEqn, N))

    for ind in range(0, len(pi), 1):
        ii = pi[ind, 0]
        jj = pj[ind, 0]

        if ii - 1 >= 0:
            if UnfilledMask_topleft[ii - 1, jj] == 0 and gradientMask[ii - 1, jj] == 0:
                UnfilledMask_topleft[ii, jj] = 0

        if jj - 1 >= 0:
            if UnfilledMask_topleft[ii, jj - 1] == 0 and gradientMask[ii, jj - 1] == 0:
                UnfilledMask_topleft[ii, jj] = 0


    for ind in range(len(pi) - 1, -1, -1):
        ii = pi[ind, 0]
        jj = pj[ind, 0]

        if ii + 1 <= imgH - 1:
            if UnfilledMask_bottomright[ii + 1, jj] == 0 and gradientMask[ii, jj] == 0:
                UnfilledMask_bottomright[ii, jj] = 0

        if jj + 1 <= imgW - 1:
            if UnfilledMask_bottomright[ii, jj + 1] == 0 and gradientMask[ii, jj] == 0:
                UnfilledMask_bottomright[ii, jj] = 0

    UnfilledMask = UnfilledMask_topleft * UnfilledMask_bottomright

    return A, b, UnfilledMask


def constructEquation(n, validN, holeMask, gradientMask, imgSrc_gx, imgSrc_gy, imgTrg, pi, pj, pind, qi, qj, qind, I, J, S, b, e_start, e_stop):

    validNeighbor = validN[:, n]

    qi_tmp = copy.deepcopy(qi)
    qj_tmp = copy.deepcopy(qj)
    qi_tmp[np.invert(validNeighbor), n] = 0
    qj_tmp[np.invert(validNeighbor), n] = 0

    if n == 0:
        HaveGrad = gradientMask[pi[:, 0], pj[:, 0]] == 0
    elif n == 2:
        HaveGrad = gradientMask[pi[:, 0], pj[:, 0] - 1] == 0
    elif n == 1:
        HaveGrad = gradientMask[pi[:, 0], pj[:, 0]] == 0
    elif n == 3:
        HaveGrad = gradientMask[pi[:, 0] - 1, pj[:, 0]] == 0

    Boundary = holeMask[qi_tmp[:, n], qj_tmp[:, n]] == 0

    valid = validNeighbor * HaveGrad * Boundary

    J_tmp = pind[valid, :]

    e_stop = e_start + len(J_tmp)
    I_tmp = np.arange(e_start, e_stop, dtype=np.float32).reshape(-1, 1)
    e_start = e_stop

    S_tmp = np.ones(J_tmp.shape, dtype=np.float32)

    if n == 0:
        b_tmp = - imgSrc_gx[pi[valid, 0], pj[valid, 0], :] + imgTrg[qi[valid, n], qj[valid, n], :]
    elif n == 2:
        b_tmp = imgSrc_gx[pi[valid, 0], pj[valid, 0] - 1, :] + imgTrg[qi[valid, n], qj[valid, n], :]
    elif n == 1:
        b_tmp = - imgSrc_gy[pi[valid, 0], pj[valid, 0], :] + imgTrg[qi[valid, n], qj[valid, n], :]
    elif n == 3:
        b_tmp = imgSrc_gy[pi[valid, 0] - 1, pj[valid, 0], :] + imgTrg[qi[valid, n], qj[valid, n], :]

    I = np.concatenate((I, I_tmp))
    J = np.concatenate((J, J_tmp))
    S = np.concatenate((S, S_tmp))
    b = np.concatenate((b, b_tmp))

    NonBoundary = holeMask[qi_tmp[:, n], qj_tmp[:, n]] == 1
    valid = validNeighbor * HaveGrad * NonBoundary

    J_tmp = pind[valid, :]

    e_stop = e_start + len(J_tmp)
    I_tmp = np.arange(e_start, e_stop, dtype=np.float32).reshape(-1, 1)
    e_start = e_stop

    S_tmp = np.ones(J_tmp.shape, dtype=np.float32)

    if n == 0:
        b_tmp = - imgSrc_gx[pi[valid, 0], pj[valid, 0], :]
    elif n == 2:
        b_tmp = imgSrc_gx[pi[valid, 0], pj[valid, 0] - 1, :]
    elif n == 1:
        b_tmp = - imgSrc_gy[pi[valid, 0], pj[valid, 0], :]
    elif n == 3:
        b_tmp = imgSrc_gy[pi[valid, 0] - 1, pj[valid, 0], :]

    I = np.concatenate((I, I_tmp))
    J = np.concatenate((J, J_tmp))
    S = np.concatenate((S, S_tmp))
    b = np.concatenate((b, b_tmp))

    S_tmp = - np.ones(J_tmp.shape, dtype=np.float32)
    J_tmp = qind[valid, n, None]

    I = np.concatenate((I, I_tmp))
    J = np.concatenate((J, J_tmp))
    S = np.concatenate((S, S_tmp))

    return I, J, S, b, e_start, e_stop
