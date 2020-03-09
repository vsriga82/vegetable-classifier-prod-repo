import logging
import pdb 
from fastai.vision import *
from sklearn.model_selection import train_test_split

def accuracy_by_category(learn):
    
    interp = ClassificationInterpretation.from_learner(learn)
    cat = interp.confusion_matrix()
    actual_category = []
    total_correct = []
    total = []
    accuracy = []
    labels = learn.data.c2i

    for i in range(len(cat)):

        actual_category.append(i)
        total_correct.append(cat[i][i])
        total.append(sum(cat[i]))
        accuracy.append(round(cat[i][i]/sum(cat[i]) * 100,2))
    
    d = {"class":list(labels.keys()),"total_correct":total_correct,"total":total,"accuracy":accuracy}
    df = pd.DataFrame(d)
    return(df)

def predict_topk_labels(learn,img,k):
    
    top5_lst = []
    labels = learn.data.c2i
    img.resize(299)
    img.show()
    pred_class,pred_idx,outputs = learn.predict(img)
    top5 = torch.topk(outputs,k)
    key_list = list(labels.keys())
    val_list = list(labels.values())
    for i in range(len(top5[1])):
        top5_lst.append(key_list[val_list.index(top5[1][i])])
    return(top5_lst)

def ranked_accuracy(preds,labels,actual_label):
    rank1=0
    rank3=0
    rank5=0
    

    for (pred,gt) in zip(preds,actual_label):
        top5 = []
        p = np.argsort(-pred)
        key_list = list(labels.keys())
        val_list = list(labels.values())
        for i in range(len(p[:5])):
            top5.append(key_list[val_list.index(p[i])])

        if gt in top5[:5]:
              rank5 += 1
        
        if gt in top5[:3]:
              rank3 += 1

        if gt in top5[:1]:
              rank1 +=1
 
    rank1 /= float(len(preds))
    rank3 /= float(len(preds))
    rank5 /= float(len(preds))
    
    return(rank1,rank3,rank5)

def get_img_path(path):
    
    actual_img_path = []
    valid_images = [".jpg",".gif",".png",".tga"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        actual_img_path.append(os.path.join(path,f))
        
    return(actual_img_path)

def get_actual_category(pat,img_path):
    category = []
    for imgpath in img_path:
        result = re.search(pat,imgpath)
        category.append(result.group())
    return(category)
