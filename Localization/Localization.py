#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:02:18 2020

@author: shivanitijare
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import img_as_float
import os
from skimage.morphology import reconstruction



# GOALS
# 1) Figure out what Sobel filter is supposed to do (particularly the #s)
# 2) Figure out how to set T1 (10% of max gradient value) (Shivani)
# 3) Actually do dilate & reconstruction (Nihal)
# 4) Figure out how to get local minima so that watershed algorithm accepts it (Silvi)



# Loads all images from folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

#Tests all images from Sample Lables
def all_images_tester():
    images = load_images_from_folder('Sample Labels')
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('INPUT', img)
        cv2.waitKey(0)

        #Get gradient using 3x3 Sobel filter
        grad = sobel_filter_method(img)
    
        #Invert gradient
        grad_inverted = cv2.bitwise_not(grad)
        cv2.imshow("Inverted Grad", grad_inverted)
    

def sobel_filter_method(img):
    # Method 1- Converts from 16S to 8U
    grad_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)  # maybe try 64F? what r these #s
    grad_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)

    # converting back to CV_8U
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # combine gradients
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    cv2.imshow('grad', grad)
    cv2.waitKey(0)
    return grad


def pre_processing(img):
    
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    # Run 3x3 Sobel Filter on image to get gradient
    grad = sobel_filter_method(img)
   
    # Change gradient to a float
    grad_float = img_as_float(grad)
   
    # Invert image --> 1-gradient
    grad_inverted = cv2.bitwise_not(grad)
    cv2.imshow('Grad Inverted', grad_inverted)
    cv2.waitKey(0)
    
    # Subtract height threshold T1 from inverted gradient (in study: 65)
    T1 = 65
    # Using OpenCV
    grad_subtracted_cv = cv2.subtract(grad_inverted, T1)
    grad_subtracted_cv_float = img_as_float(grad_subtracted_cv)

    # Reconstruction/Dialation (needs to be changed)
    h = 1
    seed = cv2.subtract( grad_subtracted_cv, 1)
    mask = grad_inverted
    grad_reconstructed = reconstruction(seed, mask, method='dilation')
    print(grad_subtracted_cv.shape)
    print(grad_subtracted_cv.dtype)
    print(grad_reconstructed.shape)
    print(grad_reconstructed.dtype) 
    
    hdome = cv2.subtract(grad_inverted, np.uint8(grad_reconstructed))
    
    grad_reconstructed_complement = cv2.bitwise_not(hdome)
    cv2.imshow('Grad Reconstructed Complement',grad_reconstructed_complement)
    cv2.waitKey(0)
    cv2.imshow("Grad Reconstructed", grad_reconstructed)
    cv2.waitKey(0)
    cv2.imshow("Hdome", hdome)
    cv2.waitKey(0)
    return grad_reconstructed_complement



def watershed_segmentation(img):
    # Get connected components from pre-processed gradient
   grad_preprocessed_8 = np.uint8(img)

   numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(grad_preprocessed_8, connectivity=8)
  
   print(labels)
   print(labels.shape)
   print(labels.dtype)
   print(stats)

   labels_to_show = labels.astype(np.uint8)
   cv2.imshow('CCs', labels_to_show)
   cv2.waitKey(0)

    # Watershed on gradient using OpenCV (Meyer) - Method 1
   grad_preprocessed_8 = cv2.cvtColor(grad_preprocessed_8, cv2.COLOR_GRAY2BGR)
   grad_watershed_1 = cv2.watershed(grad_preprocessed_8, labels)
   
   grad_watershed_to_show_1 = grad_watershed_1.astype(np.uint8)
   cv2.imshow('Watershed with Labels as Markers OpenCV', grad_watershed_to_show_1)
   cv2.waitKey(0)

   # Watershed on gradient using skimage - Method 2 *TO-DO*


   # Get complement of watershed image
   watershed_complement = cv2.bitwise_not(grad_watershed_to_show_1)
   cv2.imshow('Watershed Complement', watershed_complement)
   cv2.waitKey(0)
   return watershed_complement, labels, stats, numLabels

def size_filtering(img,labels,stats,numLabels):
 h, w = labels.shape

 T2 = 0.001*h*w

 labeledConnectedComponents = np.copy(stats)
 images = 0
 for stat in stats:
     if stat[cv2.CC_STAT_AREA] >= T2:
         images += 1
         for i in range(stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_LEFT] + stat[cv2.CC_STAT_WIDTH] - 1):
            # Top line of bounding box
            img[stat[cv2.CC_STAT_TOP], i] = 0
            # Bottom line of bounding box
            img[stat[cv2.CC_STAT_TOP] + stat[cv2.CC_STAT_HEIGHT] - 1, i] = 0
            # Left line of bounding box
           # img[i, stat[cv2.CC_STAT_LEFT]] = 0
            # Right line of bounding box
           # img[i, stat[cv2.CC_STAT_LEFT] + stat[cv2.CC_STAT_WIDTH] - 1] = 0


 print('[INFO]: Total number of connected components: ' + str(numLabels))
 print('[INFO]: Total number of images classified: ' + str(images))
 print('[INFO]: Total number of texts classified: ' + str(numLabels - images))
 cv2.imshow("partial bounding box on complement",img)
 cv2.waitKey(0)
 return img

#def text_merging(img):

#Run Localazization

pre_processed = pre_processing('Sample Labels/medical-label.jpg')
segmented, labels, stats, numLabels = watershed_segmentation(pre_processed)
filtered = size_filtering(segmented, labels, stats, numLabels)

