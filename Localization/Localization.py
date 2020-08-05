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
from skimage.morphology import reconstruction, rectangle, disk
from scipy.ndimage import binary_erosion
from skimage.color import rgb2gray
from PIL import Image
import random


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


# Tests all images from Sample Labels
def all_images_tester(folder):
    images, filenames = load_images_from_folder(folder)
    for i in range(len(images)):
        original = images[i]
        resized_original, scale, gradient_bounding_boxes, dilation_bounding_boxes = pre_processing(original)
        image_regions, text_regions = classification(gradient_bounding_boxes, dilation_bounding_boxes)
        returned_bounding_boxes, bounding_box_locations, finalImg = get_final_bounding_boxes(original, scale, image_regions)
        cv2.imshow(str(filenames[i]), finalImg)
        cv2.waitKey(0)


def get_all_symbol_aspect_ratios():
    main = "symbols/"
    folder = [os.path.join(main, folder) for folder in os.listdir(main) if not (folder.startswith('.'))]
    symbols = [os.path.join(d, f) for d in folder for f in os.listdir(d)[:1]]

    aspectratios = []
    for symbol in symbols:
        img = cv2.imread(symbol)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h,w = img.shape
            aspectratio = w/h
            if aspectratio < .75 or aspectratio > 1.25:
                aspectratios.append(aspectratio)
    return aspectratios


def canny_filter(img, sigma=0.33):
    median = np.median(img)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    edged = cv2.Canny(img, lower, upper)
    return edged


def dilation(img, dsize):
    img = rgb2gray(img)
    img = cv2.resize(img, dsize)
    th = .6
    img[img <= th] = 0
    img[img > th] = 1
    img = 1 - img
    mask = img
    seed = binary_erosion(img, disk(1.2))
    recon = reconstruction(seed, mask, 'dilation')
    return recon


def resizing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resizing all imgs to 50000-90000 pixels
    h, w = img.shape
    img_size = h * w
    target_size = 50000
    scale_factor = np.sqrt(target_size / img_size)
    dsize = (int(np.round(w * scale_factor)), int(np.round(h * scale_factor)))
    img = cv2.resize(img, dsize)
    resized_img = np.copy(img)
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)

    return img, resized_img, scale_factor, dsize


def pre_processing(original):
    img, resized_original, scale, dsize = resizing(original)
    gradient = canny_filter(img)
    dilated = dilation(original, dsize)
    gradient_bounding_boxes = segmentation(gradient)
    dilated_bounding_boxes = segmentation(dilated)
    return resized_original, scale, gradient_bounding_boxes, dilated_bounding_boxes


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

    return image_size, image_regions, text_regions


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


def MergeOverlapping(regions, loop=True, xbuffer=0, ybuffer=0):
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
        if numImgsInRegion / (numTextInRegion + numImgsInRegion + 1) > 0.2:
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


def text_merging(text_regions):
    overlapped_regions = MergeOverlapping(text_regions,xbuffer=1,ybuffer=1)
    return overlapped_regions


def classification(gradient_bounding_boxes, dilation_bounding_boxes):
    img_size, image_regions, text_regions = filtering(gradient_bounding_boxes)
    image_regions, text_regions = second_segmentation(image_regions, text_regions)
    text_regions = text_merging(text_regions)
    image_regions = MergeOverlapping(image_regions, xbuffer=-1, ybuffer=-1)

    symbolratios = get_all_symbol_aspect_ratios()
    filtered_dilation_regions = []
    for region in dilation_bounding_boxes:
        keep = False
        left, top, width, height = region[0:4]
        area = width * height
        aspectratio = width / height
        if 20 < area < .2*img_size:
            if 0.75 <= aspectratio <= 1.25:
                keep = True
            else:
                for symbolratio in symbolratios:
                    if .95 * symbolratio <= aspectratio <= 1.05 * symbolratio:
                        keep = True
        if keep:
            filtered_dilation_regions.append([left, top, width, height, area])

    if filtered_dilation_regions:
        image_regions = np.vstack((image_regions, filtered_dilation_regions))
    return image_regions, text_regions


def get_final_bounding_boxes(img, scale_factor, regions, image=True):
    finalimg = np.copy(img)
    returned_bounding_boxes = []
    bounding_box_locations = []
    for region in regions:
        left = int(np.round(region[0] / scale_factor))
        top = int(np.round(region[1] / scale_factor))
        right = left + int(np.round(region[2]) / scale_factor)
        bottom = top + int(np.round(region[3]) / scale_factor)
        returned_bounding_boxes.append(img[top:bottom + 1, left:right + 1])
        bounding_box_locations.append([left, top, right, bottom])
        if image:
            cv2.rectangle(finalimg, (left, top), (right, bottom), (128, 0, 128))
        else:
            cv2.rectangle(finalimg, (left, top), (right, bottom), (0, 255, 0))
    return returned_bounding_boxes, bounding_box_locations, finalimg


def annotateDocument(file, labels, confidence, bounding_box_locations):
    labels = []
    confidence = []
    labeled_bounding_box_locs = bounding_box_locations

    for i in range(1, len(labeled_bounding_box_locs) + 1):
        labels.append(chr(ord('@') + i%12))
        confidence.append(random.randint(10,100))

    image = Image.open(file)
    w, h = image.size
    new_width = int(1.7 * w)
    newImg = Image.new(image.mode, (new_width, h), (255, 255, 255))
    newImg.paste(image, (0, 0))
    newImg = np.asarray(newImg)

    regiondellist = []
    for i, regiona in enumerate(labeled_bounding_box_locs):
        for j, regionb in enumerate(labeled_bounding_box_locs):
            if j > i:
                merged = DoMerge(regiona, regionb, xbuffer=-1, ybuffer=-1, fortext=True)
                if merged and labels[i] == labels[j]:
                    # keep region with higher confidence
                    if confidence[j] > confidence[i]:
                        regiondellist.append(i)
    labeled_bounding_box_locs = np.delete(labeled_bounding_box_locs, regiondellist, axis=0)
    labels = np.delete(labels, regiondellist, axis=0)
    confidence = np.delete(confidence, regiondellist, axis=0)

    locations = []
    for i in range(0, len(labels)):
        color = (random.randint(1, 254), random.randint(1, 254), random.randint(1, 254))
        cv2.rectangle(newImg, (labeled_bounding_box_locs[i][0], labeled_bounding_box_locs[i][1]),
                      (labeled_bounding_box_locs[i][2], labeled_bounding_box_locs[i][3]), color, thickness=2)
        if i < len(labels) / 2:
            top = int((2 * h / len(labels)) * i)
            left = w
            bottom = top + int(h / len(labels))
            locations.append([top, bottom])
        else:
            left = int(1.35 * w)
            index = i - int(len(labels) / 2) - 1
            top, bottom = locations[index][0:2]

        cv2.rectangle(newImg, (left, top), (left + 15, bottom), color, thickness=-1)
        cv2.putText(newImg, labels[i], (left + 25, bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color,
                    1, cv2.LINE_AA)

    cv2.imshow('rectangles', newImg)
    cv2.waitKey(0)
    return newImg

# Run Localization
'''
original = cv2.imread('Sample Labels/tumblr_inline_opzh8tb2Ep1tu0keb_640.png')
resized_original, scale, gradient_bounding_boxes, dilation_bounding_boxes = pre_processing(original)
image_regions, text_regions = classification(gradient_bounding_boxes, dilation_bounding_boxes)
returned_bounding_boxes, bounding_box_locations, finalImg = get_final_bounding_boxes(original, scale, image_regions)
cv2.imshow('FINAL', finalImg)
cv2.waitKey(0)
'''
#all_images_tester('Sample Labels')
