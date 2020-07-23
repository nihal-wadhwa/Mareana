# import the necessary packages
import os
import re
import cv2
import pickle
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS
from main import segment

# set the project root directory as the static folder, you can set others.
app = Flask(__name__, static_url_path='')
CORS(app)

#function to load keras model
lb = None
model = None
def my_model():
    global lb
    global model

    # load the model and label binarizer
    print("[INFO] loading network and label binarizer...")
    model = load_model("src/saved_models/resnet.model")
    lb = pickle.loads(open("src/saved_models/resnet_lb.pickle", "rb").read())

    print("[INFO] network and label binarizer loaded!")

# function to read docs from client directory
def read_docs():
    alldocs = []
    for filename in os.listdir('sample_labels'):
        if filename.endswith('.png'):
            alldocs.append(os.path.join('sample_labels/', filename))
        else:
            continue
    return alldocs


@app.route("/classify")
def classify():
    #initialize the returned data dictionary
    data = {"success": False}
    alldocs = read_docs()
    print("[INFO] processing medical documents...")
    for doc in alldocs:
        # Symbol counter for retrieving label location
        s = 0
        # Copy image for labelling
        symbols, locations, original = segment(cv2.imread(doc))
        doc_copy = original.copy()
        
        for symbol in symbols:
            label_loc = locations[s]
            symbol_copy = symbol.copy()

            # preprocess the symbol
            symbol = cv2.resize(symbol, (32, 32))
            symbol = symbol.astype("float") / 255.0
            symbol = symbol.reshape((1, symbol.shape[0], symbol.shape[1], symbol.shape[2]))

            # make predicion on the symbol
            preds = model.predict(symbol)
            # find the class label index with the largest corresponding
		    # probability
            i = preds.argmax(axis=1)[0]
            label = lb.classes_[i]

            if (preds[0][i] >= 0.7):
			    # draw the class label + probability on the output image
                confidence = "{}_{:.0f}%".format(label, preds[0][i] * 100)
                final_doc = cv2.putText(doc_copy, confidence, label_loc, cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                (0, 0, 255), 1, cv2.LINE_AA)
                
                # save symbols to local directory for future training 
                if os.path.isdir("saved_symbols/{}".format(label)):
                    cv2.imwrite('saved_symbols/{}/{}.png'.format(label, confidence), symbol_copy)
                else:
                    os.mkdir("saved_symbols/{}".format(label))
                    cv2.imwrite('saved_symbols/{}/{}.png'.format(label, confidence), symbol_copy)
            # increment next symbol
            s += 1

        # code snippet for saving annotated document
        label = doc.split(os.path.sep)[-1].split(".")[0]
        cv2.imwrite('annotated_docs/{}.png'.format(label), final_doc)
        print("[INFO] just processed and saved a doc!")

    print("[INFO] medical documents processed!")
    data = {"success": True}
    return data
    
# if this is the main thread of execution first load the model and
# then start the server

print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
my_model()
app.run()