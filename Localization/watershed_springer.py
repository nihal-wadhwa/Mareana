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
# 1) Figure out seed/mask
# 2) Figure out upper number threshold for canny edge

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
    edges = cv2.Canny(img, 65, 800)
    cv2.imshow("Canny Gradient w 800 upper bound", edges)
    cv2.waitKey(0)
    return edges

def autocanny(img, sigma=0.33):
    median = np.median(img)
    lower = int(max(0, (1.0-sigma)*median))
    upper = int(min(255, (1.0+sigma)*median))
    print('Median:' + str(median))
    print('Lower Bound:' + str(lower))
    print('Upper Bound:' + str(upper))
    edged = cv2.Canny(img,lower,upper)
    cv2.imshow('Auto Canny with ' + str(upper) +' Bound', edged)
    cv2.waitKey(0)
    return edged

# Gets rid of whitespace - does not work on every img
def getRidOfWhiteSpaces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    ## (2) Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    ## (3) Find the max-area contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    ## (4) Crop and save it
    x, y, w, h = cv2.boundingRect(cnt)
    img = img[y:y + h, x:x + w]

    return img

def dilationreconstruction(img):
    img = cv2.imread(img)
    img = rgb2gray(img)
    dimensions = img.shape

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
    print(numLabels)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    print(markers.shape)
    print(markers.dtype)
    print(processed_img.shape)
    print(processed_img.dtype)
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(processed_img, markers)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    original_img[markers == -1] = [255, 0, 0]
    markers = np.uint8(markers)
    cv2.imshow('Watershed-markers from internet', markers)
    cv2.waitKey(0)
    cv2.imshow('Watershed original img from internet', original_img)
    cv2.waitKey(0)
    print('[INFO] {} unique segments found'.format(len(np.unique(markers)) - 1))


# Load all images from Sample Labels
images = load_images_from_folder('Sample Labels')
for img in images:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('INPUT', img)
    # cv2.waitKey(0)

    # Get gradient using 3x3 Sobel filter
    # grad = sobel_filter_method_1(img)

    # Invert gradient
    # grad_inverted = cv2.bitwise_not(grad)
    # cv2.imshow("Inverted Grad", grad_inverted)


# Load UDI sample img
img = cv2.imread('Sample Labels/medical-label.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Initialize T1
# Run 3x3 Sobel Filter on image to get gradient
#grad = sobel_filter_method_1(img)
grad = autocanny(img, sigma=0.33)



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
cv2.imshow('Grad preprocessed 8 ', grad_preprocessed_8)
cv2.waitKey(0)
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
print("Num Labels Before Watershed:" + str(numLabels))

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

print('[INFO] {} unique segments found'.format(len(np.unique(watershed_complement)) - 1))

h, w = labels.shape
image_size = h*w
print(stats)
T2 = 0.001*h*w
print(T2)
labeled_img = np.array(watershed_complement)
labeled_original_img = np.array(img)
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_GRAY2BGR)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img[grad_watershed_1 == -1] = [255,0,0]
cv2.imshow("img?", img)
cv2.waitKey(0)
labeled_original_img = cv2.cvtColor(labeled_original_img, cv2.COLOR_GRAY2BGR)
print(labeled_img.shape)
images = 0

# Arrays to store bounding boxes by location
numSections = 4
bounding_box_quad_TL = []
bounding_box_quad_TR = []
bounding_box_quad_BL = []
bounding_box_quad_BR = []

filtered_images = 0
returned_bounding_boxes = []
bounding_box_locations = []
for stat in stats:
    print(stat[cv2.CC_STAT_LEFT])
    left = stat[cv2.CC_STAT_LEFT]
    top = stat[cv2.CC_STAT_TOP]
    height = stat[cv2.CC_STAT_HEIGHT]
    width = stat[cv2.CC_STAT_WIDTH]
    area = stat[cv2.CC_STAT_AREA]
    right = left + width
    bottom = top + height
    aspectratio = width / height
    if 4 <= area <= (.75 * image_size) and width < (.75 * w) and height < (.75 * h):
        if area >= T2 or 0.75 <= aspectratio <= 1.25:  # image
            coord = (left, top, right, bottom)
            if left < w / 2 and top < h / 2:  # in top left quadrant
                if len(bounding_box_quad_TL) < 1:
                    bounding_box_quad_TL.append(coord)
                    filtered_images += 1
                    returned_bounding_boxes.append(original_img[top:bottom + 1, left:right + 1])
                    bounding_box_locations.append((left, top))
                    cv2.rectangle(labeled_img, (left, top), (right, bottom), (128, 0, 128), thickness=1)
                else:
                    for i in range(len(bounding_box_quad_TL)):
                        if left >= bounding_box_quad_TL[i][0] and top >= bounding_box_quad_TL[i][1] and right <= \
                                bounding_box_quad_TL[i][2] and bottom <= bounding_box_quad_TL[i][3]:
                            print('cc inside larger cc')
                            # bounding box is inside larger bounding box --> do nothing
                        else:
                            bounding_box_quad_TL.append(coord)
                            filtered_images += 1
                            returned_bounding_boxes.append(original_img[top:bottom + 1, left:right + 1])
                            bounding_box_locations.append((left, top))
                            cv2.rectangle(labeled_img, (left, top), (right, bottom), (128, 0, 128), thickness=1)
                cv2.imshow("Bounding boxes on original image by size & aspect ratio", labeled_img)
                cv2.waitKey(0)
            if left < w / 2 and top >= h / 2:  # in bottom left quadrant
                for i in range(len(bounding_box_quad_BL)):
                    if left >= bounding_box_quad_BL[i][0] and top >= bounding_box_quad_BL[i][1] and right <= \
                            bounding_box_quad_BL[i][2] and bottom <= bounding_box_quad_BL[i][3]:
                        print('cc inside larger cc')
                        # bounding box is inside larger bounding box --> do nothing
                    else:
                        print('hello')
                        bounding_box_quad_BL.append(coord)
                        filtered_images += 1
                        returned_bounding_boxes.append(original_img[top:bottom + 1, left:right + 1])
                        bounding_box_locations.append((left, top))
                        cv2.rectangle(labeled_img, (left, top), (right, bottom), (128, 0, 128), thickness=1)
            if left >= w / 2 and top < h / 2:  # in top right quadrant
                for i in range(len(bounding_box_quad_TR)):
                    if left >= bounding_box_quad_TR[i][0] and top >= bounding_box_quad_TR[i][1] and right <= \
                            bounding_box_quad_TR[i][2] and bottom <= bounding_box_quad_TR[i][3]:
                        print('cc inside larger cc')
                        # bounding box is inside larger bounding box --> do nothing
                    else:
                        print('hello')
                        bounding_box_quad_TR.append(coord)
                        filtered_images += 1
                        returned_bounding_boxes.append(original_img[top:bottom + 1, left:right + 1])
                        bounding_box_locations.append((left, top))
                        cv2.rectangle(labeled_img, (left, top), (right, bottom), (128, 0, 128), thickness=1)

            if left >= w / 2 and top >= h / 2:  # in bottom right quadrant
                for i in range(len(bounding_box_quad_BR)):
                    if left >= bounding_box_quad_BR[i][0] and top >= bounding_box_quad_BR[i][1] and right <= \
                            bounding_box_quad_BR[i][2] and bottom <= bounding_box_quad_BR[i][3]:
                        print('cc inside larger cc')
                        # bounding box is inside larger bounding box --> do nothing
                    else:
                        print('hello')
                        bounding_box_quad_BR.append(coord)
                        filtered_images += 1
                        returned_bounding_boxes.append(original_img[top:bottom + 1, left:right + 1])
                        bounding_box_locations.append((left, top))
                        cv2.rectangle(labeled_img, (left, top), (right, bottom), (128, 0, 128), thickness=1)
        else:  # text
            cv2.rectangle(labeled_img, (left, top), (right, bottom), (0, 255, 0), thickness=1)

print('[INFO]: Total number of connected components: ' + str(numLabels))
print('[INFO]: Total number of images classified: ' + str(images))
print('[INFO]: Total number of texts classified: ' + str(numLabels - images))
cv2.imshow("partial bounding box on original image", labeled_original_img)
cv2.waitKey(0)




