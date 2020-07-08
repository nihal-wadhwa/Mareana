from tensorflow.keras.models import load_model
import argparse
import pickle
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default=None,
    help="Path of the image we want to classify")
parser.add_argument("--model", type=str, default=None,
	help="Whether or not we should flatten the image based on the model")
parser.add_argument("--pickle", type=str, default=None,
	help="Path to pickle file")
parser.add_argument("--weights", type=str, default=None,
	help="Path to model file")
args = vars(parser.parse_args())

# load the input image and resize it to the target spatial dimensions
image = cv2.imread(args['image'])
output = image.copy()
image = cv2.resize(image, (32, 32))
# scale the pixel values to [0, 1]
image = image.astype("float") / 255.0

# check to see if we should flatten the image and add a batch
# dimension
if args['model'] == 'nnet':
	image = image.flatten()
	image = image.reshape((1, image.shape[0]))
# otherwise, we must be working with a CNN -- don't flatten the
# image, simply add the batch dimension
else:
	image = image.reshape((1, image.shape[0], image.shape[1],
		image.shape[2]))

# load the model and label binarizer
print("[INFO] loading network and label binarizer...")
model = load_model(args['weights'])
lb = pickle.loads(open(args['pickle'], "rb").read())
# make a prediction on the image
preds = model.predict(image)
# find the class label index with the largest corresponding
# probability
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]

# draw the class label + probability on the output image
text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
print(text)
cv2.putText(output, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
	(0, 0, 255), 1, cv2.LINE_AA)
# show the output image
cv2.imshow("Image", output)
print("0")
cv2.waitKey(0)
