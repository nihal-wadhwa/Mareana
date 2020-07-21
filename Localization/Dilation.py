"""
Demo using Dilation tools in OpenCV

Author: Nihal Wadhwa
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage as ndi
from scipy.ndimage import binary_erosion
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import *
import os
from skimage.morphology import reconstruction, rectangle, disk

from Localization.Localization import load_images_from_folder


def disk_1(folder):
    images, filenames = load_images_from_folder(folder)
    for i in range(len(images)):
        img = rgb2gray(images[i])
        target_size = 50000
        h, w = img.shape
        img_size = h * w
        scale_factor = np.sqrt(target_size / img_size)
        dsize = (int(np.round(w * scale_factor)), int(np.round(h * scale_factor)))
        img = cv2.resize(img, dsize)
        th = .6         #.3 if the symbols are in the background
        img[img <= th] = 0
        img[img > th] = 1
        img = 1 - img
        mask = img
        seed = binary_erosion(img, disk(1.2))
        recon = reconstruction(seed, mask, 'dilation')
        cv2.imshow(filenames[i], recon)
        cv2.waitKey(0)

def disk_image(im):
    img = rgb2gray(plt.imread(im))
    target_size = 50000
    h, w = img.shape
    img_size = h * w
    scale_factor = np.sqrt(target_size / img_size)
    dsize = (int(np.round(w * scale_factor)), int(np.round(h * scale_factor)))
    img = cv2.resize(img, dsize)
    th = .6
    img[img <= th] = 0
    img[img > th] = 1
    img = 1 - img
    mask = img
    cv2.imshow("Mask", mask)
    seed = binary_erosion(img, disk(1.2))
    recon = reconstruction(seed, mask, 'dilation')
    cv2.imshow('Output', recon)
    cv2.waitKey(0)

