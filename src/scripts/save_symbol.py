import cv2
from tensorflow.keras.models import load_model
import argparse
import pickle
import sys
import os

# function to save symbols for future training
# name each image with label + probability

def save_symbol(symbols):
    model = load_model('src/saved_models/resnet.model')
    lb = pickle.loads(open("src/saved_models/resnet_lb.pickle", "rb").read())

    for symbol in symbols:
        symbol = cv2.resize(symbol, (32, 32))
        symbol_copy = symbol.copy()
        symbol = symbol.astype("float") / 255.0
        symbol = symbol.reshape((1, symbol.shape[0], symbol.shape[1], symbol.shape[2]))

        preds = model.predict(symbol)

        i = preds.argmax(axis=1)[0]
        label = lb.classes_[i]

        confidence = "{}_{:.0f}".format(label, (preds[0][i] * 100))
        #img = cv2.imdecode(symbol, cv2.IMREAD_GRAYSCALE)

        if os.path.isdir("saved_symbols/{}".format(label)):
            cv2.imwrite('saved_symbols/{}/{}.png'.format(label, confidence), symbol_copy)
        else:
            os.mkdir("saved_symbols/{}".format(label))
            cv2.imwrite('saved_symbols/{}/{}.png'.format(label, confidence), symbol_copy)