# import the necessary packages
import os
import time
import cv2
from flask import Flask, request, jsonify, send_from_directory
#from main import segment, predict

# set the project root directory as the static folder, you can set others.
app = Flask(__name__, static_url_path='')

@app.route('/time')
def get_current_time():
    return {'time': time.time()}

@app.route("/predict")
def predict():
    alldocs = []
    for filename in os.listdir('sample_labels'):
        if filename.endswith(('.jpg', '.png', 'jpeg')):
            alldocs.append(os.path.join('sample_labels', filename))
        else:
            continue
    return {'docs': alldocs}
    
    for doc in alldocs:
        symbols, locations, original = segment(cv2.imread(img_path))
        final_doc = predict_symbols(symbols, locations, original)
        cv2.imwrite('output/', final_doc)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run()