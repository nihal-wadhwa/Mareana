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
# 1) Figure out seed/mask
# 2) Figure out upper number threshold for canny edge
# 3) Figure out left/right bounding box edges



# Loads all images from folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


def sobel_filter_method_1(img):
    # Method 1- Converts from 16S to 8U
    grad_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)  # maybe try 64F? what r these #s
    cv2.imshow('x', grad_x)
    cv2.waitKey(0)
    grad_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)
    cv2.imshow('y', grad_y)
    cv2.waitKey(0)

    # converting back to CV_8U
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    cv2.imshow('abs x', abs_grad_x)
    cv2.waitKey(0)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    cv2.imshow('abs y', abs_grad_y)
    cv2.waitKey(0)

    # combine gradients
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    cv2.imshow('grad', grad)
    cv2.waitKey(0)
    return grad


def sobel_filter_method_2(img):
    # Method 2 - Converts from 64F to 8U
    # Output dtype = cv.CV_64F. then take its abs and convert to 8U
    sobelx64f = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely64f = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobelx64f = np.absolute(sobelx64f)
    abs_sobely64f = np.absolute(sobely64f)
    abs_sobelxy = np.sqrt(abs_sobelx64f ** 2 + abs_sobely64f ** 2)
    sobel_abs_8uxy = np.uint8(abs_sobelxy)
    cv2.imshow('SOBEL_ABS_8U', sobel_abs_8uxy)
    cv2.waitKey(0)
    return sobel_abs_8uxy


def sobel_filter_method_3(img):
    # naive method - had worst results on dataset
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    # cv2.imshow('SOBEL_X',sobel_x)
    # cv2.waitKey(0)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    # cv2.imshow('SOBEL_Y',sobel_y)
    # cv2.waitKey(0)
    sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # cv2.imshow('SOBEL', sobel)
    # cv2.waitKey(0)
    return sobel

def canny_filter(img):
    edges = cv2.Canny(img, 65, 650)
    cv2.imshow("Canny Gradient w 800 upper bound", edges)
    cv2.waitKey(0)
    return edges

# Load all images from Sample Labels
images = load_images_from_folder('Sample Labels')
for img in images:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('INPUT', img)
    # cv2.waitKey(0)

    # Get gradient using 3x3 Sobel filter
    # grad = sobel_filter_method_1(img)

    # Invert gradient
    # grad_inverted = cv2.bitwise_not(grad)
    # cv2.imshow("Inverted Grad", grad_inverted)


# Load UDI sample img
img = cv2.imread('Sample Labels/UDI_label_.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Run 3x3 Sobel Filter on image to get gradient
#grad = sobel_filter_method_1(img)
grad = canny_filter(img)

# Change gradient to a float
grad_float = img_as_float(grad)
grad_inverted_float = 1 - grad_float

# Invert image --> 1-gradient
grad_inverted = cv2.bitwise_not(grad)
cv2.imshow('Grad Inverted', grad_inverted)
cv2.waitKey(0)

# Change inverted gradient to a float
# grad_inverted_float = img_as_float(grad_inverted)

# Subtract height threshold T1 from inverted gradient (in study: 65)
T1 = 65

# Using OpenCV
grad_subtracted_cv = cv2.subtract(grad_inverted, T1)
cv2.imshow('Grad Inverted Subtracted Using CV2', grad_subtracted_cv)
cv2.waitKey(0)
grad_subtracted_cv_float = img_as_float(grad_subtracted_cv)


# Using NP
# grad_subtracted_np = np.subtract(grad_inverted, T1)
# cv2.imshow('Grad Inverted Subtracted Using NP', grad_subtracted_np)
# cv2.waitKey(0)

# Morphological Reconstruction with OpenCV
# Method 1 - WORST, NO COMPLEMENT
seed = np.copy(grad_subtracted_cv)
mask = np.copy(grad_inverted)
grad_reconstructed_1 = reconstruction(seed, mask, method="dilation")
hdome1 = grad_subtracted_cv_float - grad_reconstructed_1
# cv2.imshow('Grad Reconstructed 1 Using CV', grad_reconstructed_1)
# cv2.waitKey(0)
grad_reconstructed_1_complement = cv2.bitwise_not(grad_reconstructed_1)
# cv2.imshow('Grad Reconstructed 1 Hdome1 Complement', hdome1)
# cv2.waitKey(0)

# Method 2 - BEST, BAD COMPLEMENT
seed = np.copy(grad_subtracted_cv)
seed[1:-1, 1:-1] = grad_subtracted_cv.min()
mask = grad_subtracted_cv
grad_reconstructed_2 = reconstruction(seed, mask, method='dilation')
hdome2 = grad_subtracted_cv_float - grad_reconstructed_2
#cv2.imshow('Grad Reconstructed 2 Using Grad Subtracted', grad_reconstructed_2)
#cv2.waitKey(0)
#grad_reconstructed_2_complement = cv2.bitwise_not(grad_reconstructed_2)
#cv2.imshow('Grad Reconstructed 2 Hdome2 Complement', hdome2)
cv2.waitKey(0)

# Method 3 - METHOD 2 BUT GRAY background, BEST COMPLEMENT
h = 1
seed = cv2.subtract(grad_subtracted_cv_float,0.4)
mask = grad_inverted
grad_reconstructed_3 = reconstruction(seed, mask, method='dilation')
hdome3 = cv2.subtract(grad_inverted,np.uint8(grad_reconstructed_3))
cv2.imshow('Grad Reconstructed 3 Using Grad Subtracted & H', grad_reconstructed_3)
cv2.waitKey(0)
grad_reconstructed_3_complement = cv2.bitwise_not(hdome3)
cv2.imshow('Grad Reconstructed: HDOME3', hdome3)
cv2.waitKey(0)
cv2.imshow('Grad Reconstructed Complement', grad_reconstructed_3_complement)
cv2.waitKey(0)

# Input = grad_reconstructed_3_complement, hdome3, hdome 2

# Get connected components from pre-processed gradient
grad_preprocessed_8 = np.uint8(grad_reconstructed_3_complement)

# Invert preprocessed gradient
grad_preprocessed_inverted = cv2.bitwise_not(grad_preprocessed_8)
grad_preprocessed_inverted = img_as_float(grad_preprocessed_inverted)

image_max = ndi.maximum_filter(grad_preprocessed_inverted, size=20 ,mode='constant')
coordinates = peak_local_max(grad_preprocessed_inverted, min_distance=20)
print(coordinates)
minima = np.ones(grad_preprocessed_inverted.shape)
for coordinate in coordinates:
    minima[coordinate[0], coordinate[1]] = 0
cv2.imshow('minima', minima)
cv2.waitKey(0)
minima = minima.astype(np.uint8)

numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(grad_preprocessed_8, connectivity=8)

print(labels)
print(labels.shape)
print(labels.dtype)
print(minima.dtype)


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

#cv2.imshow('Watershed Complement CCs', labels1_to_show)
#cv2.waitKey(0)
#output = cv2.connectedComponentsWithStats(watershed_complement, connectivity=8)
#        for i in range(output[0]):
#            if output[2][i][4] >= min_thresh and output[2][i][4] <= max_thresh:
#                cv2.rectangle(frame, (output[2][i][0], output[2][i][1]), (
#                    output[2][i][0] + output[2][i][2], output[2][i][1] + output[2][i][3]), (0, 255, 0), 2)
#        cv2.imshow('detection', frame)

# watershed_floodfill = np.copy(watershed_complement)
# h, w = watershed_complement.shape[:2]
# mask = np.ones((h+2, w+2), np.uint8)
# num, im, mask, rect = cv2.floodFill(watershed_floodfill, mask, (0,0),255)
# print(num)
# cv2.imshow('Floodfill', mask)
# cv2.waitKey(0)
# floodfill_inv = cv2.bitwise_not(mask)
# cv2.imshow('Inverted Floodfill', floodfill_inv)
# cv2.waitKey(0)
# floodfill_combined = watershed_complement | floodfill_inv
# cv2.imshow('Combined Floodfill', floodfill_combined)
# cv2.waitKey(0)

h, w = labels.shape
print(stats)
T2 = 0.001*h*w
print(T2)
labeledConnectedComponents = np.copy(stats)
labeled_img = np.array(watershed_complement)
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_GRAY2BGR)
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

    #for i in range(stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_LEFT] + stat[cv2.CC_STAT_WIDTH] - 1):
        # Top line of bounding box
        #watershed_complement[stat[cv2.CC_STAT_TOP], i] = 0
        # Bottom line of bounding box
        #watershed_complement[stat[cv2.CC_STAT_TOP] + stat[cv2.CC_STAT_HEIGHT] - 1, i] = 0
        # Left line of bounding box
        # watershed_complement[i, stat[cv2.CC_STAT_LEFT]] = 0
        # Right line of bounding box
        # watershed_complement[i, stat[cv2.CC_STAT_LEFT] + stat[cv2.CC_STAT_WIDTH] - 1] = 0
    if stat[cv2.CC_STAT_AREA] >= T2:
        images += 1
        cv2.rectangle(labeled_img, (left, top), (right, bottom), (128, 0, 128), thickness=1)
    else:
        cv2.rectangle(labeled_img, (left, top), (right, bottom), (0, 255, 0), thickness=1)

print('[INFO]: Total number of connected components: ' + str(numLabels))
print('[INFO]: Total number of images classified: ' + str(images))
print('[INFO]: Total number of texts classified: ' + str(numLabels - images))
cv2.imshow("partial bounding box on complement", labeled_img)
cv2.waitKey(0)




