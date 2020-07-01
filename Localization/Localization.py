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
# 1) Actually do dilate & reconstruction (Nihal)
# 2) Delete bounding boxes inside bigger ones, merge overlapping boxes (not touching ones) --> run while loop until no
#    bounding boxes overlap, redetermine parameters for size classification (T2) (Silvi/Shivani)



# Loads all images from folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    print(len(images))
    return images

#Tests all images from Sample Labels
def all_images_tester(folder):
    images = load_images_from_folder(folder)
    for img in images:
        original, pre_processed = pre_processing(img)
        # original, segmented, label, statistics, numLabel = watershed_segmentation(original, pre_processed)
        original, segmented, label, statistics, numLabel = watershed_segmentation(original, pre_processed)
        size_filtering(original, segmented, label, statistics, numLabel)



def canny_filter(img, sigma = 0.33):
    median = np.median(img)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    print('Median:' + str(median))
    print('Lower Bound:' + str(lower))
    print('Upper Bound:' + str(upper))
    edged = cv2.Canny(img, lower, upper)
    cv2.imshow('Auto Canny with ' + str(upper) + ' Bound', edged)
    cv2.waitKey(0)
    return edged

def pre_processing(img):
    #img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    # Run 3x3 Canny Filter on image to get gradient
    grad = canny_filter(img,sigma=0.33)
   
    # Invert image --> 1-gradient
    # grad_inverted = cv2.bitwise_not(grad)
    # cv2.imshow('Grad Inverted', grad_inverted)
    # cv2.waitKey(0)

    # Subtract height threshold T1 from inverted gradient (in study: 65)
    # T1 = 65
    # Using OpenCV
    # grad_subtracted_cv = cv2.subtract(grad_inverted, T1)
    # grad_subtracted_cv_float = img_as_float(grad_subtracted_cv)

    # Reconstruction/Dilation (needs to be changed)
    # seed = cv2.subtract(grad_subtracted_cv_float, 0.4)
    # mask = grad_inverted
    # grad_reconstructed = reconstruction(seed, mask, method='dilation')
    # hdome = cv2.subtract(grad_inverted, np.uint8(grad_reconstructed))
    # grad_reconstructed_complement = cv2.bitwise_not(hdome)

    # cv2.imshow('Grad Reconstructed 3 Using Grad Subtracted & H', grad_reconstructed)
    # cv2.waitKey(0)
    # cv2.imshow('Grad Reconstructed: HDOME', hdome)
    # cv2.waitKey(0)
    # cv2.imshow('Grad Reconstructed Complement', grad_reconstructed_complement)
    # cv2.waitKey(0)
    return img, grad



def watershed_segmentation(original_img, processed_img):
    # Get connected components from pre-processed gradient
   grad_preprocessed = np.uint8(processed_img)

   numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(grad_preprocessed, connectivity=8)
   print(numLabels)

    # Watershed on gradient using OpenCV (Meyer) - Method 1
   grad_preprocessed = cv2.cvtColor(grad_preprocessed, cv2.COLOR_GRAY2BGR)
   grad_watershed = cv2.watershed(grad_preprocessed, labels)

   grad_watershed_to_show = grad_watershed.astype(np.uint8)
   cv2.imshow('Watershed with Labels as Markers OpenCV', grad_watershed_to_show)
   cv2.waitKey(0)

   # Watershed on gradient using skimage - Method 2 *TO-DO*

   # Get complement of watershed image
   watershed_complement = cv2.bitwise_not(grad_watershed_to_show)
   cv2.imshow('Watershed Complement', watershed_complement)
   cv2.waitKey(0)
   return original_img, watershed_complement, labels, stats, numLabels


def size_filtering(original_img, segmented_img,labels,stats,numLabels):
 h, w = labels.shape

 T2 = 0.001*h*w

 labeled_img = np.array(segmented_img)
 labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_GRAY2BGR)
 labeled_original_img = np.array(original_img)
 labeled_original_img = cv2.cvtColor(labeled_original_img, cv2.COLOR_GRAY2BGR)
 print(labeled_img.shape)
 images = 0
 for stat in stats:
     print(stat[cv2.CC_STAT_LEFT])
     left = stat[cv2.CC_STAT_LEFT]
     top = stat[cv2.CC_STAT_TOP]
     height = stat[cv2.CC_STAT_HEIGHT]
     width = stat[cv2.CC_STAT_WIDTH]
     area = stat[cv2.CC_STAT_AREA]
     right = left + width
     bottom = top + height
     if area >= T2:
         images += 1
         cv2.rectangle(labeled_img, (left, top), (right, bottom), (128, 0, 128), thickness=1)
         cv2.rectangle(labeled_original_img, (left, top), (right, bottom), (128, 0, 128), thickness=1)

     else:
         cv2.rectangle(labeled_img, (left, top), (right, bottom), (0, 255, 0), thickness=1)
         cv2.rectangle(labeled_original_img, (left, top), (right, bottom), (0, 255, 0), thickness=1)

 print('[INFO]: Total number of connected components: ' + str(numLabels))
 print('[INFO]: Total number of images classified: ' + str(images))
 print('[INFO]: Total number of texts classified: ' + str(numLabels - images))
 cv2.imshow("partial bounding box on original image", labeled_original_img)
 cv2.waitKey(0)

#def text_merging(img):

#Run Localization

#original, pre_processed = pre_processing('Sample Labels/fda-fictitious-medical-device-udi-identifier.jpg')
#original, segmented, label, statistics, numLabel = watershed_segmentation(original, pre_processed)
#watershed_test(original, pre_processed)
#size_filtering(original, segmented, label, statistics, numLabel)
all_images_tester('Sample Labels')

