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

def annotateDocument(original, labels, bounding_box_locations):
	#labels = []
	labeled_bounding_box_locs = bounding_box_locations

	#for i in range(1, len(labeled_bounding_box_locs) + 1):
	#        labels.append(chr(ord('@') + i))

	image = original
	dim = image.shape()
	w = dim[1]
	h = dim[0]
	new_width = int(1.55 * w)
	label_box = np.ones((h, new_width - w)) * 255
	newImg = np.hstack((image, label_box))
	#newImg = Image.new(image.mode, (new_width, h), (255, 255, 255))
	#newImg.paste(image, (0, 0))
	#newImg = np.asarray(newImg)
	#cv2.imshow('pil img', newImg)
	#cv2.waitKey(0)

	locations = []
	for i in range(0, len(labels)):
		color = (random.randint(1, 254), random.randint(1, 254), random.randint(1, 254))
		cv2.rectangle(newImg, (labeled_bounding_box_locs[i][0], labeled_bounding_box_locs[i][1]),
                      (labeled_bounding_box_locs[i][2], labeled_bounding_box_locs[i][3]), color, thickness=2)
		if (i < len(labels) / 2):
			top = int((2 * h / len(labels)) * i)
			left = w
			bottom = top + int(h / len(labels))
			locations.append([top, bottom])
		else:
			left = int(1.225 * w)
			index = i - int(len(labels) / 2) - 1
			top, bottom = locations[index][0:2]

		cv2.rectangle(newImg, (left, top), (left + 15, bottom), color, thickness=-1)
		cv2.putText(newImg, labels[i], (left + 35, top + int((bottom - top) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 
				1, cv2.LINE_AA)

	#cv2.imshow('rectangles', newImg)
	#cv2.waitKey(0)
	return newImg

# function to run the symbol prediction process
def predict_symbols(symbols, locations, original):
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
			label_str = label + str(preds[0][i])
			symbol_labels.append(label_str)
			# draw the class label + probability on the output image
			#text = "{}: {:.0f}%".format(label, preds[0][i] * 100)		
			#symbol_img = cv2.putText(symbol_img, text, label_loc, cv2.FONT_HERSHEY_SIMPLEX, 0.3,
			#	(0, 0, 255), 1, cv2.LINE_AA)
		# Increment next symbol
		s += 1
	
	symbol_img = annotateDocument(original, symbol_labels, symbol_locs)

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
