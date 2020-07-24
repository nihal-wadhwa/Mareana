import os
import math
import sys
import argparse
import numpy as np
import argparse
import cv2
import pickle

# CHANGE TO SYSTEM PATH TO LOCALIZATION SCRIPT
sys.path.insert(1, '/Users/dhruv/Desktop/Document-Symbol-Classification/')
from src.scripts.save_symbol import save_symbol
from src.scripts.predict import predict_symbols
from Localization.Localization import pre_processing, classification, get_final_bounding_boxes

'''
# extract the arguments 
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
'''

def segment(original):
    resized_original, scale, gradient_bounding_boxes, dilation_bounding_boxes = pre_processing(original)
    image_regions, text_regions = classification(gradient_bounding_boxes, dilation_bounding_boxes)
    returned_bounding_boxes, bounding_box_locations, final_img = get_final_bounding_boxes(original, scale, image_regions)

    return returned_bounding_boxes, bounding_box_locations, final_img
    

def predict(symbols, locations, original):
    final_doc = predict_symbols(symbols, locations, original)
    return final_doc

def collect(symbols):
    save_symbol(symbols)

'''
# call functions based on --task values
if args.task == 'segment':
    segment(cv2.imread(doc_path))

elif args.task == 'predict':
    symbols, locations, original = segment(cv2.imread(img_path))
    predict(symbols, locations, original)

elif args.task == 'collect':
    symbols, locations, original = segment(cv2.imread(img_path))
    save_symbol(symbols)
'''


#symbols, locations, original = segment(cv2.imread(img_path))
#collect(symbols)
