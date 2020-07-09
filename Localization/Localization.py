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
# 1) Actually do dilate & reconstruction (Nihal)
# 2) Delete bounding boxes inside bigger ones, merge overlapping boxes (not touching ones) --> run while loop until no
#    bounding boxes overlap, redetermine parameters for size classification (T2) (Silvi/Shivani)



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
    # print('Median:' + str(median))
    # print('Lower Bound:' + str(lower))
    # print('Upper Bound:' + str(upper))
    edged = cv2.Canny(img, lower, upper)
    cv2.imshow('Auto Canny with ' + str(upper) + ' Bound', edged)
    cv2.waitKey(0)
    return edged

def pre_processing(img):
    img = cv2.imread(img)
    original_img = np.copy(img)
    img = rgb2gray(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    # Run 3x3 Canny Filter on image to get gradient
    # grad = canny_filter(img,sigma=0.33)
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
    seed = binary_erosion(img, rectangle(1, int(.015 * width)))  # 1,4 for fda, 1,30 for UDI
    recon = reconstruction(seed, mask, 'dilation')
    cv2.imshow('Output', recon)
    cv2.waitKey(0)

    return original_img, recon

def watershed_test(original_img, processed_img):

    img = np.copy(original_img)
    processed_img = cv2.bitwise_not(processed_img)
    ret, thresh = cv2.threshold(processed_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    # opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(closing, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 3)

    # Threshold
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    numLabels, markers, stats, centroids = cv2.connectedComponentsWithStats(sure_fg)
    # print(numLabels)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    # print(markers.shape)
    # print(markers.dtype)
    # print(processed_img.shape)
    # print(processed_img.dtype)
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(processed_img, markers)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    original_img[markers == -1] = [255, 0, 0]
    markers = np.uint8(markers)
    cv2.imshow('Watershed-markers from internet', markers)
    cv2.waitKey(0)
    cv2.imshow('Watershed original img from internet', original_img)
    cv2.waitKey(0)
    # print('[INFO] {} unique segments found'.format(len(np.unique(markers)) - 1))

def watershed_segmentation(original_img, processed_img):
    # Get connected components from pre-processed gradient
   grad_preprocessed = np.uint8(processed_img)

   numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(grad_preprocessed, connectivity=8)
   # print(numLabels)

    # Watershed on gradient using OpenCV (Meyer) - Method 1
   grad_preprocessed = cv2.cvtColor(grad_preprocessed, cv2.COLOR_GRAY2BGR)
   grad_watershed = cv2.watershed(grad_preprocessed, labels)

   grad_watershed_to_show = grad_watershed.astype(np.uint8)
   #cv2.imshow('Watershed with Labels as Markers OpenCV', grad_watershed_to_show)
   #cv2.waitKey(0)

   # Watershed on gradient using skimage - Method 2 *TO-DO*

   # Get complement of watershed image
   watershed_complement = cv2.bitwise_not(grad_watershed_to_show)
   cv2.imshow('Watershed Complement', watershed_complement)
   cv2.waitKey(0)
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

original, pre_processed = pre_processing('Sample Labels/journal.pone.0165002.g003.png')
original, segmented, label, statistics, numLabel = watershed_segmentation(original, pre_processed)
original_img, labeled_img, bounding_box_array = filtering(original, segmented, label, statistics, numLabel)

#all_images_tester('Sample Labels')

