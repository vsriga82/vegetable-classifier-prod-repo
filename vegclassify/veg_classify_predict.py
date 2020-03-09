#----------------------------------
#import required packages\libraries
#----------------------------------

from fastai.vision import *
#import fastai.vision
#from fastai.vision import data
from fastai.callbacks import CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
import pandas as pd
import numpy as np
from PIL import Image
import os, os.path
import warnings
import logging
warnings.filterwarnings("ignore")
#----------------------------------------------------------------------------
#Required for windows machine to specify the path where packages are located.
#----------------------------------------------------------------------------
import sys
sys.path.append('C:\\Users\\sriga\\Desktop\\ML\\Springboard\\vegetable-classifier-prod\\vegclassify')
import build_data
import custom_metrics
import veg_classify_prep_data
import argparse



def single_predictions(imgpath,learn):
    
    img = open_image(imgpath)
    img.resize(299)
    learn.predict(img)
    top5 = custom_metrics.predict_topk_labels(learn,img,5)
    logging.info(top5)

    return(top5)

def batch_predictions(learn,test_path):

    preds,y = learn.get_preds(ds_type=DatasetType.Test)
    logging.info(len(preds))

    #---------------------------------------------------------------
    # Groud truth labels need to be extracted from test image names.
    # Below fuction returns the test image path in a list
    #---------------------------------------------------------------
    img_path = []
    img_path = custom_metrics.get_img_path(test_path)

    #------------------------------------------------------------------------------------
    # Using regular expression extract the label name from image path. The result will be 
    # in format 'carrot.jpg'
    #------------------------------------------------------------------------------------
    pat = r'([^/\d\_]+)\.jpg$'
    actual_category = []
    actual_category = custom_metrics.get_actual_category(pat,img_path)

    #----------------------------------------------------
    #Extract just the label name. i.e. drop the .jpg part
    #----------------------------------------------------
    actual_label = []
    for img in actual_category:
        re = img.split(".",1)[0]
        actual_label.append(re)

    #-------------------------------------------------------------------------------------------
    # Call the function ranked_accuracy which will retrun rank 1, rank 2 and rank 3 predictions
    #-------------------------------------------------------------------------------------------
    labels = learn.data.c2i
    rank1,rank3,rank5 = custom_metrics.ranked_accuracy(preds,labels,actual_label)
    return(rank1*100,rank3*100,rank5*100)

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Main program starting.")

    ap = argparse.ArgumentParser()

    ap.add_argument("-p","--prediction",required=True,help="single or batch prediction")
    ap.add_argument("-ml","--modelpath",required=True,help="path where model pickle file is stored")
    ap.add_argument("-i","--img",help="img for single prediction")
    ap.add_argument("-t","--testpath",help="path where test images are stored")

    args = vars(ap.parse_args())

    # Single\Batch predictions
    if  args["prediction"] == 'single':
        imgpath = args["img"]
        logging.info(imgpath)
        learn = load_learner(args["modelpath"])
        top_5_predictions = single_predictions(imgpath,learn)
        logging.info(top_5_predictions)
    elif args["prediction"] == 'batch':
        test_data = ImageList.from_folder(args["testpath"])
        learn = load_learner(args["modelpath"],test=test_data)
        rank1,rank2,rank3 = batch_predictions(learn,args["testpath"])
        logging.info(rank1)
        logging.info(rank2)
        logging.info(rank3)

if __name__ == '__main__':
    main()
