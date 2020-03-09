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
import argparse


def build_data_for_training(imgpath,classes):

    logging.info("Starting to build the data for training")
#------------------------------------------------------------------------------------------
#validate whether an image is valid i.e. can it be opened as valid image. If not delete it.
#------------------------------------------------------------------------------------------
    #build_data.validate_images(imgpath,classes)

#------------------------------------------------------------------------------------------
# Preprocess and build data. Here label is derived from the folder name. 
# Unfortunately the split between training and testing is 80:20 on overall data and not at 
# each class level. Hence classes with less images may not be picked for validation
# reference - https://forums.fast.ai/t/stratified-labels-sampling/28002/8
#------------------------------------------------------------------------------------------
    
    databunch = build_data.build_data_from_folder(imgpath,split=0.2,size=224,bs=16)

#------------------------------------------------------------------------------------------
# As a workaround, we need to build the data somewhere else outside and pass it to fast ai 
# library. I have used scikit learn library to create the train test split in a stratified 
# way. It has two steps 1) create a dataframe from databunch 2) Pass that dataframe to 
# scikit learn train_test_split and with an option to strtify the results. The result will
# be trainX, validX, trainY, validY dataframes
#-------------------------------------------------------------------------------------------
    result = build_data.build_df_from_databunch(databunch)
    (trainX, validX, trainY, validY) = train_test_split(result['x'],result['y'],stratify=result['y'],test_size=0.2, random_state=42)

#-------------------------------------------------------------------------------------------
# Now we have stratified train and validation data as dataframe. But fast ai needs them as 
# databunch. Inorder to build the databunch out of data frame, fast ai needs to know which 
# records needs to be considered for training v/s validation. We already have built seperate
# train and valid dataframes in above step. We just need to add one more label to dataframe 
# to indicate which record need to be used for validation. The final_df is a dataframe that
# has both training and validation data combined, but now with a label indicating which rows
# are validatoin records v/s training records. 
#-------------------------------------------------------------------------------------------
    final_df = build_data.build_stratified_data(trainX,trainY,validX,validY)

#-------------------------------------------------------------------------------------------
# The final step is to build the fast ai databunch from the dataframe.
#-------------------------------------------------------------------------------------------
    final_databunch = build_data.build_data_from_df(final_df,imgpath,size=224,bs=16)
    logging.info(final_databunch.c)
    logging.info(len(final_databunch.train_ds))
    logging.info(len(final_databunch.valid_ds))

    logging.info("Data preperation is complete.")
    return(final_databunch)


def train_model(data,epoch=4,maxlr=0.003,mod=models.resnet34):

    logging.info("Starting to train the model.")   
    top_3_accuracy = partial(top_k_accuracy, k=3)
    learn = cnn_learner(data, mod, metrics=[error_rate,accuracy,top_3_accuracy,top_k_accuracy],callback_fns=[CSVLogger])
    learn.fit_one_cycle(epoch,maxlr)
    #path = Path('..\\results')
    accuracy_by_cat = custom_metrics.accuracy_by_category(learn)
    accuracy_by_cat.to_csv()
    logging.info("Completed training, returning back to main.")
    return(learn)


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Main program starting.")

    ap = argparse.ArgumentParser()

    ap.add_argument("-p","--path",required=True,help="path where training images are saved")
    ap.add_argument("-e","--epoch",required=True,help="number of epoch to be used for training")
    ap.add_argument("-lr","--maxlr",required=True,help="Max learning rate")
    ap.add_argument("-ml","--model",required=True,help="model to be used for training")

    args = vars(ap.parse_args())

#------------------------------------------------
# Define the images source path and valid classes
#------------------------------------------------
    #imgpath = Path('C:\\Users\\sriga\\Desktop\\ML\\Springboard\\vegetable-classifier-prod\\data\\images')
    imgpath = Path(args["path"])
    logging.info(imgpath)
    classes = ['ash gourd','asparagus','bamboo shoot','basil','beans','beetroot','bitter gourd','black raddish','bottle gourd','brinjal','broccoli','cabbage','capsicum','carrot',
           'cauliflower','celeriac','chayote','chilli','chinese artichokes','cluster beans','coconut','colocasia','coriander leaves','corn','cucumber','curry leaves','dill',
           'drumstick','dulse','elephant yam','fenugreek leaves','fiddleheads','flat beans','garlic','ginger','gooseberry','green mango','ivy gourd','kohlrabi','lemon','lime',
           'long beans','lotus root','mint','mushroom','nopal','oca','okra','onion','peas','plantain','plantain flower','plantain stem','potato','pumpkin','ramps','red chilli',
           'red raddish','ridge gourd','romanesco','shallots','snake gourd','sweet potato','tapioca','tomato','turnip','white onion','white raddish','yam', 'zuchini']


    train_databunch = build_data_for_training(imgpath,classes)
    

    epoch = int(args["epoch"])
    maxlr = float(args["maxlr"])
    mod = args["model"]
    
    valid_models = { "resnet18": models.resnet18, 
                     "resnet34": models.resnet34, 
                     "resnet50": models.resnet50,
                     "resnet101": models.resnet101,
                     "resnet152": models.resnet152,
                     "squeezenet1_0": models.squeezenet1_0,
                     "squeezenet1_1": models.squeezenet1_1,
                     "densenet121": models.densenet121,
                     "densenet169": models.densenet169,
                     "densenet201": models.densenet201,
                     "densenet161": models.densenet161,
                     "vgg16_bn": models.vgg16_bn, 
                     "vgg19_bn": models.vgg19_bn,
                     "alexnet": models.alexnet,
                     "unet": models.unet}

    value = valid_models.get(args["model"],None)
    if value is not None:
        mod = value
    else:
        logging.error("Model not supported by fastai")

    logging.info(epoch)
    logging.info(maxlr)
    logging.info(mod)

    learner = train_model(train_databunch,epoch,maxlr,mod)

    learner.path = Path('..\\model')
    learner.export()

if __name__ == '__main__':
    main()
