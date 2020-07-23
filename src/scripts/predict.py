import cv2
from tensorflow.keras.models import load_model
import argparse
import pickle
import sys
import os
from PIL import Image

'''

# CHANGE TO SYSTEM PATH TO LOCALIZATION SCRIPT
sys.path.insert(1, '~/mareana/Desktop/Document-Symbol-Classification/Localization')
from Localization import pre_processing, segmentation, filtering, second_segmentation, get_final_bounding_boxes

# extract the arguments 
parser = argparse.ArgumentParser(description=
'Arguments to run prediction script')

parser.add_argument('--image', type=str, 
	default=None,
    help="Path of the image we want to classify")

parser.add_argument("--model", type=str, 
	default=None,
	help="Type of model")

args = vars(parser.parse_args())
'''
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



def annotateDocument(file, labels, confidence, bounding_box_locations):
	labeled_bounding_box_locs = bounding_box_locations

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

	#cv2.imshow('rectangles', newImg)
	#cv2.waitKey(0)
	
	return newImg

# function to run the symbol prediction process
def predict_symbols(symbols, locations, original, file):
	symbol_img = original.copy()
	model = 'resnet'
	# load the model and label binarizer
	print("[INFO] loading network and label binarizer...")
	model = load_model("src/saved_models/resnet.model")
	lb = pickle.loads(open("src/saved_models/resnet_lb.pickle", "rb").read())

	# Symbol counter for retrieving label location
	s = 0 
	symbol_locs = []
	symbol_labels = []
	symbol_conf = []
	# make a prediction on the images
	for image in symbols:
		output = image.copy()
		image = cv2.resize(image, (32, 32))
		image = image.astype("float") / 255.0
		# check to see if we should flatten the image and add a batch
		# dimension
		if model == 'nnet':
				image = image.flatten()
				image = image.reshape((1, image.shape[0]))
		# otherwise, we must be working with a CNN -- don't flatten the
		# image, simply add the batch dimension
		else:
				image = image.reshape((1, image.shape[0], image.shape[1],
						image.shape[2]))

		preds = model.predict(image)
		# find the class label index with the largest corresponding
		# probability
		i = preds.argmax(axis=1)[0]
		label = lb.classes_[i]
		
		if (preds[0][i] >= 0.5):
			symbol_locs.append(locations[s])
			symbol_labels.append(label)
			symbol_conf.append(preds[o][i])
			# draw the class label + probability on the output image
			#text = "{}: {:.0f}%".format(label, preds[0][i] * 100)		
			#symbol_img = cv2.putText(symbol_img, text, label_loc, cv2.FONT_HERSHEY_SIMPLEX, 0.3,
			#	(0, 0, 255), 1, cv2.LINE_AA)
		# Increment next symbol
		s += 1
	
	symbol_img = annotateDocument(file, symbol_labels, symbol_conf, symbol_locs)

	return symbol_img	
	# show the output image
	#cv2.imshow("Labeled Image", symbol_img)
	#cv2.waitKey(0)

	# Save labeled documents for reference (change path)
	#cv2.imwrite('boxed_img.png', labeled_img)

'''
# run localization methods
original = cv2.imread(args['image'])
resized_original, scale, pre_processed = pre_processing(original)
bounding_boxes = segmentation(pre_processed)
image_regions, text_regions = filtering(bounding_boxes)
image_regions, text_regions = second_segmentation(image_regions, text_regions)
#text_regions = text_merging(text_regions)
returned_bounding_boxes, bounding_box_locations, final_img = get_final_bounding_boxes(original, scale, image_regions)
predict_symbol(returned_bounding_boxes, bounding_box_locations,original)
'''
