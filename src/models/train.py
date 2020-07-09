import os
import PIL
import matplotlib.pyplot as plt
import numpy as np
import cv2
import re
import pickle
import argparse
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import GlorotUniform as glorot_uniform
from tensorflow.keras import backend as K
from small_vggnet import SmallVGGNet
from initial_neural_net import InitialNN
from resnet import ResNet50, identity_block, conv_block
# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None,
                    help="""
                    Type of model to train
                    """)
args = vars(parser.parse_args())

# Read images from dataset
data, labels = [], []
main = "../../val/"
folder = [os.path.join(main, folder) for folder in os.listdir(main)]
symbols = [os.path.join(d, f) for d in folder for f in os.listdir(d)]

for symbol in symbols:
    image = cv2.imread(symbol)
    if args['model'] == 'nnet':
        image = cv2.resize(image, (32, 32)).flatten()
    elif args['model'] == 'vggnet':
        image = cv2.resize(image, (32, 32))
    elif args['model'] == 'resnet':
        image = cv2.resize(image, (32, 32))
    data.append(image)
    label = symbol.split(os.path.sep)[-2].split(".")[0]
    label = re.sub('\_\d*', '', label)
    labels.append(label)


# Preprocessing - uniform dimensions? image enhancement?
# scale the raw pixel intensities to the range [0, 1]

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.20, random_state=42)


# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# consult_ifu [1 0 0 0]
# do_not_resterilize [0 1 0 0]
# keep_dry [0 0 1 0]
# sterile [0 0 0 1]


# Initialize model type
if args['model'] == 'nnet':
    model = InitialNN.build(lb)

elif args['model'] == 'vggnet':
    # initialize our VGG-like Convolutional Neural Network
    model = SmallVGGNet.build(width=32, height=32, depth=3,
                              classes=len(lb.classes_))
elif args['model'] == 'resnet':
    model = ResNet50(input_shape=(32, 32, 3), classes=len(lb.classes_))

print(labels)
print('lb.classes: ', lb.classes_)
# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.01
EPOCHS = 1
BS = 32
# compile the model using SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the neural network
H = model.fit(x=trainX, y=trainY, validation_data=(testX, testY),
              epochs=EPOCHS, batch_size=32)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.show()

# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")

model.save("../output/{}.model".format(args['model']), save_format="h5")
f = open("../output/{}_lb.pickle".format(args['model']), "wb")

f.write(pickle.dumps(lb))
f.close()


