import logging
import pdb 
from fastai.vision import *
from sklearn.model_selection import train_test_split
import pytest
import sys
sys.path.append('C:\\Users\\sriga\\Desktop\\ML\\Springboard\\vegetable-classifier-prod\\vegclassify')
import build_data
import custom_metrics
import warnings
warnings.filterwarnings("ignore")

from fastai.callbacks import CSVLogger


imgpath = Path('C:\\Users\\sriga\\Desktop\\ML\\Springboard\\vegetable-classifier-prod\\data\\images')
databunch = build_data.build_data_from_folder(imgpath,split=0.2,size=224,bs=16)
result = build_data.build_df_from_databunch(databunch)
(trainX, validX, trainY, validY) = train_test_split(result['x'],result['y'],stratify=result['y'],test_size=0.2, random_state=42)
final_df = build_data.build_stratified_data(trainX,trainY,validX,validY)
final_databunch = build_data.build_data_from_df(final_df,imgpath,size=224,bs=16)

top_3_accuracy = partial(top_k_accuracy, k=3)
learn = cnn_learner(final_databunch, models.resnet34, metrics=[error_rate,accuracy,top_3_accuracy,top_k_accuracy],callback_fns=[CSVLogger])
learn.fit_one_cycle(1,0.003)
learn.export()

accuracy_by_cat = custom_metrics.accuracy_by_category(learn)
impath = Path('C:\\Users\\sriga\\Desktop\\ML\\Springboard\\vegetable-classifier-prod\\data\\test\\18_beetroot.jpg')
img = open_image(impath)
img.resize(299)
img = final_databunch.train_ds[0][0]
pred_class,pred_idx,outputs = learn.predict(img)
top5 = custom_metrics.predict_topk_labels(learn,img,5)

testpath = Path('C:\\Users\\sriga\\Desktop\\ML\\Springboard\\Vegetable Classifier\\Dataset\\unittest')
test_data = ImageList.from_folder(testpath)
learn = load_learner(imgpath,test=test_data)

preds,y = learn.get_preds(ds_type=DatasetType.Test)

img_path = []
img_path = custom_metrics.get_img_path(testpath)

pat = r'([^/\d\_]+)\.jpg$'
actual_category = []
actual_category = custom_metrics.get_actual_category(pat,img_path)
actual_label = []

for img in actual_category:
    re = img.split(".",1)[0]
    actual_label.append(re)
labels = learn.data.c2i
rank1,rank3,rank5 = custom_metrics.ranked_accuracy(preds,labels,actual_label)



def test_build_data_from_folder():
    
    assert databunch.c == 70

def test_build_df_from_databunch():
    
    assert len(trainX) == round((len(result) * 80 )/100)
    assert len(validX) == round((len(result) * 20 )/100)

def test_build_stratified_data():
      
    assert len(final_df[final_df['is_valid'] == True]['y'].value_counts()) == 70

def test_build_data_from_df():

    assert final_databunch.c == 70

def test_accuracy_by_category():

    assert len(accuracy_by_cat) == 70

def test_predict_topk_labels():

    assert len(top5) == 5
    
def test_ranked_accuracy():

    assert rank5*100 >= 70

    
    