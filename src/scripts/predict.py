import cv2
from tensorflow.keras.models import load_model
import argparse
import pickle
import sys
import os

'''

# CHANGE TO SYSTEM PATH TO LOCALIZATION SCRIPT
sys.path.insert(1, '/Users/dhruv/Desktop/Document-Symbol-Classification/Localization')
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
	# make a prediction on the images
	for image in symbols:
		label_loc = locations[s]
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
			# draw the class label + probability on the output image
			text = "{}: {:.0f}%".format(label, preds[0][i] * 100)		
			symbol_img = cv2.putText(symbol_img, text, label_loc, cv2.FONT_HERSHEY_SIMPLEX, 0.3,
				(0, 0, 255), 1, cv2.LINE_AA)
		# Increment next symbol
		s += 1

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