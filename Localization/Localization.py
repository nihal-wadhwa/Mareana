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
from skimage.morphology import reconstruction, rectangle
from scipy.ndimage import binary_erosion
from skimage.color import rgb2gray





# GOALS
# 1) Somehow mark bounding boxes inside bigger ones, maybe merge overlapping boxes? (not touching ones) -->
#  run while loop until no bounding boxes overlap (Silvi/Shivani)



# Loads all images from folder
def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames


#Tests all images from Sample Labels
def all_images_tester(folder):
    images, filenames = load_images_from_folder(folder)
    for i in range(len(images)):
        cv2.imshow("Input: " + filenames[i], images[i])
        original, pre_processed = pre_processing(images[i])
        original, segmented, label, statistics, numLabel = watershed_segmentation(original, pre_processed)
        filtering(original, segmented, label, statistics, numLabel)


def canny_filter(img, sigma = 0.33):
    median = np.median(img)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    edged = cv2.Canny(img, lower, upper)
    # cv2.imshow('Auto Canny with ' + str(upper) + ' Bound', edged)
    # cv2.waitKey(0)
    return edged

def pre_processing(img):
    img = cv2.imread(img)
    original_img = np.copy(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = rgb2gray(img)

    # Run 3x3 Canny Filter on image to get gradient
    grad = canny_filter(img,sigma=0.33)

    # dimensions = img.shape

    # height and width of image
    # height = img.shape[0]
    # width = img.shape[1]
    # aspectratio = width / height
    # size = width * height
    # print("Aspect Ratio: " + str(aspectratio))
    # print('Size: ' + str(size))

    # th = 0.6
    # img[img <= th] = 0
    # img[img > th] = 1
    # img = 1 - img
    # cv2.imshow('Mask', img)
    # cv2.waitKey(0)

    # mask = img
    # seed = binary_erosion(img, rectangle(2, int(.015 * width)))  # 1,4 for fda, 1,30 for UDI (norm 0.015)
    # recon = reconstruction(seed, mask, 'dilation')
    # cv2.imshow('Output', recon)
    # cv2.waitKey(0)

    return original_img, grad


def watershed_segmentation(original_img, processed_img):
    # Get connected components from pre-processed gradient
   grad_preprocessed = np.uint8(processed_img)
   numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(grad_preprocessed, connectivity=8)

    # Watershed on gradient using OpenCV (Meyer)
   grad_preprocessed = cv2.cvtColor(grad_preprocessed, cv2.COLOR_GRAY2BGR)
   grad_watershed = cv2.watershed(grad_preprocessed, labels)
   grad_watershed_to_show = grad_watershed.astype(np.uint8)

   # Get complement of watershed image
   watershed_complement = cv2.bitwise_not(grad_watershed_to_show)
   # cv2.imshow('Watershed Complement', watershed_complement)
   # cv2.waitKey(0)
   return original_img, watershed_complement, labels, stats, numLabels


def filtering(original_img, segmented_img,labels,stats,numLabels):
    h, w = labels.shape

    T2 = 0.001*h*w

    labeled_img = np.array(original_img)
    filtered_images = 0
    returned_bounding_boxes = []
    for stat in stats:
        left = stat[cv2.CC_STAT_LEFT]
        top = stat[cv2.CC_STAT_TOP]
        height = stat[cv2.CC_STAT_HEIGHT]
        width = stat[cv2.CC_STAT_WIDTH]
        area = stat[cv2.CC_STAT_AREA]
        right = left + width
        bottom = top + height
        aspectratio = width / height
        if area >= 4:
            if area >= T2 or 0.75 <= aspectratio <= 1.25:  # image
                filtered_images += 1
                returned_bounding_boxes.append(original_img[top:bottom + 1, left:right + 1])
                cv2.rectangle(labeled_img, (left, top), (right, bottom), (128, 0, 128), thickness=1)

            else:  # text
                cv2.rectangle(labeled_img, (left, top), (right, bottom), (0, 255, 0), thickness=1)

    print('[INFO]: Total number of connected components: ' + str(numLabels))
    print('[INFO]: Total number of images classified by size: ' + str(filtered_images))
    print('[INFO]: Total number of texts classified by size: ' + str(numLabels - filtered_images))
    cv2.imshow("Bounding boxes on original image by size & aspect ratio", labeled_img)
    cv2.waitKey(0)
    # Returns original image, labeled image with bounding boxes, & array of matrices of bounding boxes
    return original_img, labeled_img, returned_bounding_boxes


#def text_merging(img):


#Run Localization

original, pre_processed = pre_processing('Sample Labels/fda-fictitious-medical-device-udi-identifier.jpg')
original, segmented, label, statistics, numLabel = watershed_segmentation(original, pre_processed)
original_img, labeled_img, bounding_box_array = filtering(original, segmented, label, statistics, numLabel)

#all_images_tester('Sample Labels')

