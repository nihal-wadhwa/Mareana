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
from skimage.morphology import reconstruction, rectangle

img = rgb2gray(plt.imread('Sample Labels/375_250-udi_sample.jpg'))

dimensions = img.shape

# height and width of image
height = img.shape[0]
width = img.shape[1]


th = 0.6
img[img <= th] = 0
img[img > th] = 1
img = 1 - img
cv2.imshow('Mask', img)
cv2.waitKey(0)

mask = img
seed = binary_erosion(img, rectangle(1,int(.015*width))) #1,4 for fda, 1,30 for UDI
recon = reconstruction(seed, mask, 'dilation')
cv2.imshow('Output', recon)
cv2.waitKey(0)
