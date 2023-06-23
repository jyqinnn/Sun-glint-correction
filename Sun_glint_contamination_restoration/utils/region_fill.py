import numpy as np
import cv2
from scipy import sparse
from scipy.sparse.linalg import spsolve


def regionfill(I, mask, factor=1.0):
    if np.count_nonzero(mask) == 0:
        return I.copy()
    resize_mask = cv2.resize(
        mask.astype(float), (0, 0), fx=factor, fy=factor) > 0
    resize_I = cv2.resize(I.astype(float), (0, 0), fx=factor, fy=factor)
    maskPerimeter = findBoundaryPixels(resize_mask)
    regionfillLaplace(resize_I, resize_mask, maskPerimeter)
    resize_I = cv2.resize(resize_I, (I.shape[1], I.shape[0]))
    resize_I[mask == 0] = I[mask == 0]
    return resize_I


def findBoundaryPixels(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    maskDilated = cv2.dilate(mask.astype(float), kernel)
    return (maskDilated > 0) & (mask == 0)


def regionfillLaplace(I, mask, maskPerimeter):
    height, width = I.shape
    rightSide = formRightSide(I, maskPerimeter)

    maskIdx = np.where(mask)

    rightSide = rightSide[maskIdx]

    grid = -np.ones((height, width))
    grid[maskIdx] = range(0, maskIdx[0].size)
    grid = padMatrix(grid)
    gridIdx = np.where(grid >= 0)

    i = np.arange(0, maskIdx[0].size)
    j = np.arange(0, maskIdx[0].size)

    numNeighbors = computeNumberOfNeighbors(height, width)
    s = numNeighbors[maskIdx]

    for direction in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        neighbors = grid[gridIdx[0] + direction[0], gridIdx[1] + direction[1]]
        index = (neighbors >= 0)
        i = np.concatenate((i, grid[gridIdx[0][index], gridIdx[1][index]]))
        j = np.concatenate((j, neighbors[index]))
        s = np.concatenate((s, -np.ones(np.count_nonzero(index))))

    D = sparse.coo_matrix((s, (i.astype(int), j.astype(int)))).tocsr()
    sol = spsolve(D, rightSide)
    I[maskIdx] = sol
    return I


def formRightSide(I, maskPerimeter):
    height, width = I.shape
    perimeterValues = np.zeros((height, width))
    perimeterValues[maskPerimeter] = I[maskPerimeter]
    rightSide = np.zeros((height, width))

    rightSide[1:height - 1, 1:width - 1] = (
        perimeterValues[0:height - 2, 1:width - 1] +
        perimeterValues[2:height, 1:width - 1] +
        perimeterValues[1:height - 1, 0:width - 2] +
        perimeterValues[1:height - 1, 2:width])

    rightSide[1:height - 1, 0] = (
        perimeterValues[0:height - 2, 0] + perimeterValues[2:height, 0] +
        perimeterValues[1:height - 1, 1])

    rightSide[1:height - 1, width - 1] = (
        perimeterValues[0:height - 2, width - 1] +
        perimeterValues[2:height, width - 1] +
        perimeterValues[1:height - 1, width - 2])

    rightSide[0, 1:width - 1] = (
        perimeterValues[1, 1:width - 1] + perimeterValues[0, 0:width - 2] +
        perimeterValues[0, 2:width])

    rightSide[height - 1, 1:width - 1] = (
        perimeterValues[height - 2, 1:width - 1] +
        perimeterValues[height - 1, 0:width - 2] +
        perimeterValues[height - 1, 2:width])

    rightSide[0, 0] = perimeterValues[0, 1] + perimeterValues[1, 0]
    rightSide[0, width - 1] = (
        perimeterValues[0, width - 2] + perimeterValues[1, width - 1])
    rightSide[height - 1, 0] = (
        perimeterValues[height - 2, 0] + perimeterValues[height - 1, 1])
    rightSide[height - 1, width - 1] = (perimeterValues[height - 2, width - 1] +
                                        perimeterValues[height - 1, width - 2])
    return rightSide


def computeNumberOfNeighbors(height, width):

    numNeighbors = np.zeros((height, width))
    numNeighbors[1:height - 1, 1:width - 1] = 4
    numNeighbors[1:height - 1, (0, width - 1)] = 3
    numNeighbors[(0, height - 1), 1:width - 1] = 3
    numNeighbors[(0, 0, height - 1, height - 1), (0, width - 1, 0,
                                                  width - 1)] = 2
    return numNeighbors


def padMatrix(grid):
    height, width = grid.shape
    gridPadded = -np.ones((height + 2, width + 2))
    gridPadded[1:height + 1, 1:width + 1] = grid
    gridPadded = gridPadded.astype(grid.dtype)
    return gridPadded
