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
# 1) Find universal parameter for dilation (nihal)
# 2) Make new image with REF & LOT symbols together --> hard to separate in docs


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
        original, label, statistics, numLabel = segmentation(original, pre_processed)
        original_img, text_labeled_img, text_regions, image_regions = filtering(original, label, statistics, numLabel)
        original_img, labeled_img, text_regions, image_regions, bounding_box_array, bounding_box_locations = fix_bounding_boxes(image_regions, text_regions, original_img, text_labeled_img)


def canny_filter(img, sigma = 0.33):
    median = np.median(img)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    edged = cv2.Canny(img, lower, upper)
    return edged


def pre_processing(img):
    img = cv2.imread(img)
    original_img = np.copy(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Run 3x3 Canny Filter on image to get gradient
    grad = canny_filter(img,sigma=0.33)

    return original_img, grad


def segmentation(original_img, processed_img):
    # Get connected components from pre-processed gradient
   grad_preprocessed = np.uint8(processed_img)
   numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(grad_preprocessed, connectivity=8)
   return original_img, labels, stats, numLabels


def filtering(original_img,labels,stats,numLabels):
    h, w = labels.shape
    image_size = h*w
    T2 = 0.001*h*w
    text_labeled_img = np.array(original_img)
    image_regions = []
    text_regions = []
    for stat in stats:
        left, top, width, height, area = stat[0:5]
        right = left + width
        bottom = top + height
        aspectratio = width / height
        if 4 <= area <= (.75 * image_size) and width < (.75 * w) and height < (.75 * h):
            if area >= T2 or 0.75 <= aspectratio <= 1.25:  # image
                image_regions.append([left, top, width, height, area])
            else:  # text
                cv2.rectangle(text_labeled_img, (left, top), (right, bottom), (0, 255, 0), thickness=1)
                text_regions.append([left, top, width, height, area])

    cv2.imshow("Text bounding boxes on original image by size & aspect ratio", text_labeled_img)
    cv2.waitKey(0)
    return original_img, text_regions, text_regions, image_regions, text_labeled_img


def DoMerge(regiona, regionb, xbuffer=0, ybuffer=0, fortext=False):
    """Input: two regions containing bounding box info [x, y, width, height, area]
    Output: False if the regions should not be merged, or (x,y,width,height) for the new region if they should
    Keywords:
    xbuffer: int buffer region to merge if the x boundaries are within this value
    ybuffer: int buffer region to merge if the y boundaries are within this value
    Purpose: Check if regiona and regionb overlap, if so, return their combined bounding box
    """
    if all([regiona[i] == regionb[i] for i in range(4)]):
        return (regiona[0], regiona[1], regiona[2], regiona[3])
    x1a, y1a, wa, ha = regiona[0:4]
    x1b, y1b, wb, hb = regionb[0:4]
    x2a = x1a + wa + xbuffer
    y2a = y1a + ha + ybuffer
    x2b = x1b + wb
    y2b = y1b + hb
    x1a = x1a - xbuffer
    y1a = y1a - ybuffer
    x1b = x1b
    y1b = y1b

    x_left = max(x1a, x1b)
    y_top = max(y1a, y1b)
    x_right = min(x2a, x2b)
    y_bottom = min(y2a, y2b)
    if x_right < x_left or y_bottom < y_top:
        return False
    elif fortext:
        return True
    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # compute the area of both AABBs
    bb1_area = (x2a - x1a) * (y2a - y1a)
    bb2_area = (x2b - x1b) * (y2b - y1b)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    x1f = min(x1a + xbuffer, x1b)
    x2f = max(x2a - xbuffer, x2b)  # -xbuffer
    y1f = min(y1a + ybuffer, y1b)
    y2f = max(y2a - ybuffer, y2b)  # -ybuffer
    wf = x2f - x1f
    hf = y2f - y1f
    return (x1f, y1f, wf, hf)


def MergeOverlapping(regions, loop=True, xbuffer=1, ybuffer=1):
    """Input: regions containing list of bounding box info [x, y, width, height, area]
    Output: new list of regions with the merged regions
    Keywords:
    loop: boolean if True, then loop the merge until you don't merge any regions, otherwise do it once
    xbuffer: int buffer region to merge if the x boundaries are within this value
    ybuffer: int buffer region to merge if the y boundaries are within this value
    Purpose: Take in a list of bounding boxes, loop through and check if they are overlapping within buffer
    if so, merge them and reduce the number of boxes.  Iterate and return the updated list of merged boxes
    """
    delta = -1
    numregions = len(regions)
    while delta != 0:
        numregions = len(list(regions))
        dellist = []
        tempregions = regions.copy()
        for i, regiona in enumerate(tempregions):
            for j, regionb in enumerate(tempregions):
                if j > i:
                    merged = DoMerge(regiona, regionb, xbuffer=xbuffer, ybuffer=ybuffer)
                    if merged != False:
                        tempregions[i][0:4] = merged
                        dellist.append(j)
        for index in sorted(set(dellist), reverse=True):
            tempregions = np.delete(tempregions, index, axis=0)
        regions = tempregions.copy()
        delta = numregions - len(regions)
        if loop == False:
            delta = 0
    return regions

def fix_bounding_boxes(image_regions, text_regions, original_img, labeled_img):
    image_labeled_img = np.copy(original_img)
    filtered_images = 0
    returned_bounding_boxes = []
    bounding_box_locations = []

    regions = MergeOverlapping(image_regions)
    for region in regions:
        left = region[0]
        top = region[1]
        right = left + region[2]
        bottom = top + region[3]
        filtered_images += 1
        returned_bounding_boxes.append(original_img[top:bottom + 1, left:right + 1])
        bounding_box_locations.append((left, top))
        cv2.rectangle(labeled_img, (left, top), (right, bottom), (128, 0, 128), thickness=1)
        cv2.rectangle(image_labeled_img, (left, top), (right, bottom), (128, 0, 128), thickness=1)

    cv2.imshow('Labeled Img with Merged Bounding Boxes', labeled_img)
    cv2.waitKey(0)
    print('[INFO]: Total number of images classified: ' + str(filtered_images))

    return original_img, labeled_img, text_regions, image_regions, returned_bounding_boxes, bounding_box_locations


def text_merging(original_img, labeled_img, text_regions, image_regions):
    delta = -1
    returned_bounding_boxes = []
    bounding_box_locations = []
    numimgregions = len(image_regions)
    while delta != 0:
        numimgregions = len(list(image_regions))
        imgdellist = []
        textdellist = []
        tempimgregions = image_regions.copy()
        temptextregions = text_regions.copy()
        for i, regioni in enumerate(tempimgregions):
            numTextInRegion = 0
            for j, regiont in enumerate(temptextregions):
                if j > i:
                    merged = DoMerge(regioni, regiont, xbuffer=2, ybuffer=2,fortext=True)
                    if merged:
                        numTextInRegion += 1
                        textdellist.append(j)
            if numTextInRegion / (numTextInRegion + 1) >= .80:
                #make img text
                imgdellist.append(i)
                cv2.rectangle(original_img, (regioni[0], regioni[1]), (regioni[0] + regioni[2], regioni[1] + regioni[3]),
                              (0, 255, 0), thickness=1)
        for index in sorted(set(textdellist), reverse=True):
            temptextregions = np.delete(temptextregions, index, axis=0)
        delta = 0
    for index in sorted(set(imgdellist), reverse=True):
        tempimgregions = np.delete(tempimgregions, index, axis=0)
    regions = tempimgregions.copy()
    cv2.imshow('Text Merged Img', original_img)
    cv2.waitKey(0)
    for region in regions:
        left = region[0]
        top = region[1]
        right = left + region[2]
        bottom = top + region[3]
        returned_bounding_boxes.append(original_img[top:bottom + 1, left:right + 1])
        bounding_box_locations.append((left, top))

    return regions, returned_bounding_boxes, bounding_box_locations



# Run Localization

original, pre_processed = pre_processing('Sample Labels/fda-fictitious-medical-device-udi-identifier.jpg')
original, label, statistics, numLabel = segmentation(original, pre_processed)
original_img, text_labeled_img, text_regions, image_regions, text_labeled_img = filtering(original, label, statistics, numLabel)
original_img, labeled_img, text_regions, image_regions, bounding_box_array, bounding_box_locations = fix_bounding_boxes(image_regions, text_regions, original_img, text_labeled_img)
regions, returned_bounding_boxes, bounding_box_locations = text_merging(original_img, labeled_img,text_regions,image_regions)
#all_images_tester('Sample Labels')

