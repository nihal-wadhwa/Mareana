import os
import sys
import argparse
import numpy as np
import argparse
import cv2
import pickle
from tensorflow.python.keras import models

# CHANGE TO SYSTEM PATH TO LOCALIZATION SCRIPT
sys.path.insert(1, '/Users/dhruv/Desktop/Document-Symbol-Classification/Localization')
from Localization import pre_processing, segmentation, filtering, second_segmentation, drawBoundingBoxes, get_final_bounding_boxes


#extract the arguments 
parser = argparse.ArgumentParser(description=
'Segment a document, fit the classifier, evaluate accuracy or predict class of symbol')

parser.add_argument('--task', type=str, default='pass',
                    help="""
                    task to perform: 
                    segment-->process document and draw bounding boxes
                    predict-->­­­classify bounded symbols in the document
                    collect-->­­­save extracted symbols with labels
                    """)

parser.add_argument('--seg', type=str, default=None,
                    help="""
                    Path of the image when we want to perform document segmentation
                    """)

parser.add_argument('--pred', type=str, default=None,
                    help="""
                    Path of the image that needs to be labeled
                    """)

parser.add_argument('--save', type=str, default=None,
                    help="""
                    Path of the image that needs symbol extraction
                    """)

args = parser.parse_args()

#checking the format of given arguments
if args.task not in ['segment', 'predict', 'collect']:
    print('Task not supported!')
    args.task = 'pass'

if args.task == 'segment':    
    if os.path.exists(args.seg):
        doc_path = args.seg
    else:
        print('Unknown path!')
        args.task = 'pass'


if args.task == 'predict':    
    if os.path.exists(args.pred):
        img_path = args.pred
    else:
        print('Unknown path!')
        args.task = 'pass'

if args.task == 'collect':    
    if os.path.exists(args.save):
        img_path = args.save
    else:
        print('Unknown path!')
        args.task = 'pass'



def segment(doc):
    pre_processed = pre_processing(doc)
    bounding_boxes = segmentation(pre_processed)
    image_regions, text_regions = filtering(bounding_boxes)
    image_regions, text_regions = second_segmentation(image_regions, text_regions)
    returned_bounding_boxes, bounding_box_locations = get_final_bounding_boxes(doc, image_regions)
    image_labeled_img = drawBoundingBoxes(doc, image_regions, (128,0,128))
    labeled_img = drawBoundingBoxes(image_labeled_img, text_regions, (0,255,0))

    if args.task == 'segment':
        cv2.imshow('labels after second segmentation', labeled_img)
        cv2.waitKey(0)
    else:
        return returned_bounding_boxes, bounding_box_locations, labeled_img


def predict(symbols, locations, original):
    text = {}
    model = models.load_model('src/output/vggnet.model')
    lb = pickle.loads(open("src/output/vggnet_lb.pickle", "rb").read())
    s = 0 # Symbol counter for retrieving label location
    # make a prediction on the images
    symbol_img = original.copy()

    for image in symbols:
        label_loc = locations[s]
        image = cv2.resize(image, (64, 64))
        image = image.astype("float") / 255.0
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        
        # make model preictions
        preds = model.predict(image)

        # find the class label index with largest probability
        i = preds.argmax(axis=1)[0]
        label = lb.classes_[i]

        if (preds[0][i] >= 0.7):
            # draw the class label + probability
            text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
            print(text)
            symbol_img = cv2.putText(symbol_img, text, label_loc, cv2.FONT_HERSHEY_SIMPLEX, 0.3,
			    (0, 0, 255), 1, cv2.LINE_AA)

        # Increment next symbol
        s += 1

    cv2.imshow("Image", symbol_img)
    cv2.waitKey(0)

def collect(symbols)


# call functions based on --task values
if args.task == 'segment':
    segment(cv2.imread(doc_path))

elif args.task == 'predict':
    symbols, locations, original = segment(cv2.imread(img_path))
    predict(symbols, locations, original)

elif args.task == 'collect':
    symbols, locations, original = segment(cv2.imread(img_path))
    collect(symbols)
