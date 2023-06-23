import numpy as np
import cv2
import torch


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def f1_score(self):
        p = np.diag(self.confusion_matrix)/np.sum(self.confusion_matrix, axis=1) #axis=1:TP+FP
        r = np.diag(self.confusion_matrix)/np.sum(self.confusion_matrix, axis=0) #axis=0:TP+FN
        f1 = 2*(p*r)/(p+r)
        F1 = np.nanmean(f1)
        return F1

    def Pixel_Accuracy_Mean(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        # Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        # Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(IoU)
        return IoU,MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        for gti, pri in zip(gt_image, pre_image):
            self.confusion_matrix += self._generate_matrix(gti, pri)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
