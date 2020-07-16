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
# 1) Figure out dilation (nihal)

# 2) Resize images to certain size (50K? 70K?) and then run code on it with x/ybuffer=0 or 1 && return bounding box on ORIGINAL img
# 4) secondary merging of all regions?

# Draws bounding boxes onto image
# Green: (0,255,0); Purple: (128,0,128)
def drawBoundingBoxes(img, regions, color):
    drawed = np.copy(img)
    for region in regions:
        left = region[0]
        top = region[1]
        right = left + region[2]
        bottom = top + region[3]
        cv2.rectangle(drawed, (left, top), (right, bottom), color, thickness=1)
    return drawed

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
        original = images[i]
        pre_processed = pre_processing(original)
        bounding_boxes = segmentation(pre_processed)
        image_regions, text_regions = filtering(bounding_boxes)
        image_regions, text_regions = second_segmentation(image_regions, text_regions)
        returned_bounding_boxes, bounding_box_locations = get_final_bounding_boxes(original, image_regions)


def canny_filter(img, sigma=0.33):
    median = np.median(img)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    edged = cv2.Canny(img, lower, upper)
    return edged

def pre_processing(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Run 3x3 Canny Filter on image to get gradient
    grad = canny_filter(img,sigma=0.33)

    return grad


def segmentation(processed_img):
    # Get connected components from pre-processed gradient
   grad_preprocessed = np.uint8(processed_img)
   numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(grad_preprocessed, connectivity=8)
   return stats


def filtering(stats):
    # Filters between image and text bounding boxes using set parameters
    w,h = stats[0][2:4]
    image_size = h*w
    T2 = 0.001*h*w
    image_regions = []
    text_regions = []
    for stat in stats:
        left, top, width, height, area = stat[0:5]
        aspectratio = width / height
        if 4 <= area <= (.75 * image_size) and width < (.75 * w) and height < (.75 * h):
            if area >= T2 or 0.75 <= aspectratio <= 1.25:  # image
                image_regions.append([left, top, width, height, area])
            else:  # text
                text_regions.append([left, top, width, height, area])

    return image_regions, text_regions



def DoMerge(regiona, regionb, xbuffer=1, ybuffer=1, fortext=False):
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


def MergeOverlapping(regions, loop=True, xbuffer=11, ybuffer=11):
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


#overlap img regions first - done
#from the larger img regions, see how many smaller img regions are inside of it
#if its a lot, get rid of the larger img region and keep the smaller ones inside of it

def second_segmentation(image_regions, text_regions):
    all_image_regions = image_regions
    overlapped_image_regions = MergeOverlapping(image_regions)

    temp_text_regions = np.copy(text_regions)
    temp_all_image_regions = np.copy(all_image_regions)
    temp_overlapped_regions = np.copy(overlapped_image_regions)

    imgdellist = []
    imgaddlist = []
    textdellist = []
    textaddlist = []

    for i, regiona in enumerate(temp_overlapped_regions):
        numImgsInRegion = 0
        numTextInRegion = 0
        temp_imgaddlist = []
        for j, regionb in enumerate(temp_all_image_regions):
            if j > i:
                merged = DoMerge(regiona, regionb, xbuffer=0, ybuffer=0, fortext=True)
                if merged:
                    numImgsInRegion += 1
                    temp_imgaddlist.append(regionb)
        for k, regionc in enumerate(temp_text_regions):
            if k > i:
                merged = DoMerge(regiona, regionc, xbuffer=0, ybuffer=0, fortext=True)
                if merged:
                    numTextInRegion += 1
                    textdellist.append(k)
        # gets rid of overlapped bounding box when too many imgs were inside it
        if numImgsInRegion / (numTextInRegion + numImgsInRegion + 1) > 0.9:
            imgdellist.append(i)
            imgaddlist.extend(temp_imgaddlist)
        # changes img overlapped bounding box to text bounding box
        elif numTextInRegion / (numTextInRegion + numImgsInRegion + 1) > 0.5:
            imgdellist.append(i)
            textaddlist.append(regiona)

    # Adding and deleting from image/text regions lists
    overlapped_image_regions = np.delete(temp_overlapped_regions, imgdellist, axis=0)
    text_regions = np.delete(temp_text_regions, textdellist, axis=0)
    for region in imgaddlist:
        overlapped_image_regions = np.insert(overlapped_image_regions, 0, region, axis=0)
    for region in textaddlist:
        text_regions = np.insert(text_regions, 0, region, axis=0)

    return overlapped_image_regions, text_regions

#def text_merging(image_regions, text_regions):
    #regions = np.insert(text_regions, 0, image_regions, axis=0)
    #overlapped_regions = MergeOverlapping(regions)
    #return overlapped_regions

def get_final_bounding_boxes(img, image_regions):
    returned_bounding_boxes = []
    bounding_box_locations = []
    for region in image_regions:
        left = region[0]
        top = region[1]
        right = left + region[2]
        bottom = top + region[3]
        returned_bounding_boxes.append(img[top:bottom + 1, left:right + 1])
        bounding_box_locations.append((left, top))
    return returned_bounding_boxes, bounding_box_locations

# Run Localization

original = cv2.imread('Sample Labels/journal.pone.0165002.g003.png')
pre_processed = pre_processing(original)
bounding_boxes = segmentation(pre_processed)
image_regions, text_regions = filtering(bounding_boxes)
image_regions, text_regions = second_segmentation(image_regions, text_regions)

image_labeled_img = drawBoundingBoxes(original, image_regions, (128,0,128))
labeled_img = drawBoundingBoxes(image_labeled_img, text_regions, (0,255,0))
cv2.imshow('labels after second segmentation', labeled_img)
cv2.waitKey(0)

returned_bounding_boxes, bounding_box_locations = get_final_bounding_boxes(original, image_regions)

#all_images_tester('Sample Labels')

