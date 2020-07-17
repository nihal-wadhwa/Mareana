from flask import Flask, request, jsonify
import pickle
import cv2
from flask_cors import CORS
from main import predict, segment
import numpy as np
from PIL import Image

# initialize our Flask application and the Keras model
app = Flask(__name__)
CORS(app)

@app.route('/classify', methods=['POST'])
def classify():
    # initialize the returned data dictionary
    data = {"success": False}
    if request.method == 'POST':
        if request.files.get('img'):

            data =request.files['file']
            filename = secure_filename(file.filename) # save file 
            filepath = os.path.join(app.config['imgdir'], filename);
            file.save(filepath)
            image = cv2.imread(filepath)

            bounded_image = segment(image)

            data["predictions"] = []
            #classify the image
            preds = predict(bounded_image)
            data["predictions"].append(preds)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run()