import cv2
from tensorflow.keras.models import load_model
import argparse
import pickle
import sys
import os
# CHANGE TO SYSTEM PATH TO LOCALIZATION SCRIPT
sys.path.insert(1, '/home/jampher/Document-Symbol-Classification/Localization')
from Localization import pre_processing, watershed_segmentation, filtering

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default=None,
    help="Path of the image we want to classify")
parser.add_argument("--model", type=str, default=None,
	help="Type of model")
args = vars(parser.parse_args())
original, pre_processed = pre_processing(args['image'])
original, segmented, label, statistics, numLabel = watershed_segmentation(original, pre_processed)
original_img, labeled_img, bounding_box_array = filtering(original, segmented, label, statistics, numLabel)

# load the model and label binarizer
print("[INFO] loading network and label binarizer...")
model = load_model("../output/{}.model".format(args['model']))
lb = pickle.loads(open("../output/{}_lb.pickle".format(args['model']), "rb").read())

# make a prediction on the images
for image in bounding_box_array:
	output = image.copy()
	image = cv2.resize(image, (32, 32))
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

	# draw the class label + probability on the output image
	text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
	print(text)
	#cv2.putText(output, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
	#	(0, 0, 255), 1, cv2.LINE_AA)
	# show the output image
	#cv2.imshow("Image", output)
	#cv2.waitKey(0)
