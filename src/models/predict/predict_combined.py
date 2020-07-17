import cv2
from tensorflow.keras.models import load_model
import argparse
import pickle
import sys
import os
# CHANGE TO SYSTEM PATH TO LOCALIZATION SCRIPT
sys.path.insert(1, '/home/nikhil/mareana/Document-Symbol-Classification/Localization')
from Localization import pre_processing, segmentation, filtering, second_segmentation, get_final_bounding_boxes

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default=None,
    help="Path of the image we want to classify")
parser.add_argument("--model", type=str, default=None,
	help="Type of model")
args = vars(parser.parse_args())

original = cv2.imread(args['image'])
resized_original, scale, pre_processed = pre_processing(original)
bounding_boxes = segmentation(pre_processed)
image_regions, text_regions = filtering(bounding_boxes)
image_regions, text_regions = second_segmentation(image_regions, text_regions)

image_bounding_boxes, image_bounding_box_locations = get_final_bounding_boxes(original, scale, image_regions)
text_bounding_boxes, text_bounding_box_locations = get_final_bounding_boxes(original, scale, text_regions, image=False)

symbol_img = original.copy()

# load the model and label binarizer
print("[INFO] loading network and label binarizer...")
model = load_model("../../output/{}.model".format(args['model']))
lb = pickle.loads(open("../../output/{}_lb.pickle".format(args['model']), "rb").read())

s = 0 # Symbol counter for retrieving label location
# make a prediction on the images
for image in image_bounding_boxes:
	label_loc = image_bounding_box_locations[s]
	output = image.copy()
	image = cv2.resize(image, (64, 64))
	image = image.astype("float") / 255.0
	# check to see if we should flatten the image and add a batch
	# dimension
	if args["model"] == 'nnet':
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
		text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
		print(text)
	
		symbol_img = cv2.putText(symbol_img, text, label_loc, cv2.FONT_HERSHEY_SIMPLEX, 0.3,
			(0, 0, 255), 1, cv2.LINE_AA)
	# show the output image
	#cv2.imshow("Image", output)
	#cv2.waitKey(0)
	
	# Increment next symbol
	s += 1

# Save images for reference (change path)
# path = 'home/nikhil/mareana/mareana-repo/src/models/predict/images'
#cv2.imwrite('boxed_img.png', labeled_img)
cv2.imwrite('labeled_img.png', symbol_img)
