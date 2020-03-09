import logging
import pdb 
from fastai.vision import *
from sklearn.model_selection import train_test_split

#pdb.set_trace()
def build_data_from_folder(imgpath,split=0.2,size=224,bs=16):
    '''
    Build a Databunch object from a folder. 

    Step 1: Build train and valid LabelLists from a folder indicated in path imgpath. Both train and valid LabelLists are made up of x = ImageList
    and y = CategoryLists i.e. groud truth labels
    Step 2: Apply standard transformations on LabelLists and build a ImageDataBunch which is required by Fastai for training.

    Parameters
    ----------
    imgpath - Path of the folder where images are stored
    split - The percentage of split between train and validation. Default to 0.2
    size - Size of the image to be transformed. Default is 224
    bs - Batch size. Default is 16

    Returns
    ----------
    Databunch
    
    '''
    
    logging.info("starting to build data from folder.")
    np.random.seed(42)
    src = (ImageList.from_folder(imgpath)
                    .split_by_rand_pct(split)
                    .label_from_folder())
    
    data = (src.transform(tfms=get_transforms(),size=224)
            .databunch(bs=bs).normalize(imagenet_stats))
    
    
    logging.info("returning the data built from folder.")
    return(data)
    

def build_df_from_databunch(data):

    '''
    Build a dataframe from a databunch received. 

    Step 1: Databunch contains LabelLIsts 'train_ds' and 'valid_ds', which are made up of x: ImageLists y: CategoryLists. First step is to 
    convert these to dataframe by invoking to_df() function.  
    Step 2: Combine the train and valid dataframes to a single data frame, sort it by 'y' (Category) and return the dataframe 

    Parameters
    ----------
    data - ImageDataBunch
    
    Returns
    ----------
    DataFrame
    
    '''
    logging.info("starting to build dataframe from databunch.")
    dt_train = data.train_ds.to_df()
    dt_valid = data.valid_ds.to_df()
    frames = [dt_train,dt_valid]
    result = pd.concat(frames)
    result = result.sort_values(by=['y'])
    logging.info("returning the dataframe back to main.")
    return(result)


def build_stratified_data(trainX,trainY,validX,validY):

    '''
    Build a dataframe from a databunch received. 

    Step 1: Input received are train and valid information as pandas series. Convert them to valid and train dataframe. 
    Step 2: In valid dataframe add a new column 'is_valid' with value 'True' and in train dataframe add a new colum 'is_valid' 
    with value 'False'
    Step 3: Combine the valid and train dataframe with new is_valid column and return it back

    Parameters
    ----------
    trainX - pandas series of images (e.g. lemon/00000072.jpeg)
    trainY - ground truth labels (e.g. lemon)
    validX - pandas series of images (e.g. carrot/00000192.jpg)
    validY - grouth truth labels (e.g. carrot)

    Returns
    ----------
    DataFrame
    
    '''
    logging.info("starting to build stratified classes.")
    valid_X_df = pd.DataFrame(validX)
    valid_Y_df = pd.DataFrame(validY)
    valid_df = pd.concat([valid_X_df,valid_Y_df],axis=1)
    valid_df['is_valid'] = True

    
    train_X_df = pd.DataFrame(trainX)
    train_Y_df = pd.DataFrame(trainY)
    train_df = pd.concat([train_X_df,train_Y_df],axis=1)
    train_df['is_valid'] = False
    
    final_data_df = pd.concat([train_df,valid_df],axis=0)
    final_data_df = final_data_df.sort_values(by=['y'])
    final_data_df = final_data_df.reset_index(drop=True)
    
    logging.info("returning the final dataframe.")
    return(final_data_df)


def build_data_from_df(final_df,imgpath,size=224,bs=16):

    '''
    Build a Databunch object from a dataframe. 

    Step 1: Build train and valid LabelLists from a folder indicated in path imgpath. Both train and valid LabelLists are made up of x = ImageList
    and y = CategoryLists i.e. groud truth labels
    Step 2: Apply standard transformations on LabelLists and build a ImageDataBunch which is required by Fastai for training.

    Parameters
    ----------
    final_df - Dataframe which has X, Y and is_valid which is an indication to fastai as to which of the rows need to be considered as 
    validation v/s training record
    size - Size of the image to be transformed. Default is 224
    bs - Batch size. Default is 16

    Returns
    ----------
    Databunch
    
    '''
    logging.info("starting to build databunch from df.")
    np.random.seed(42)
    src = (ImageList.from_df(final_df,path=imgpath)
                    .split_from_df(col='is_valid')
                    .label_from_df(cols='y')
          )
    data = (src.transform(tfms=get_transforms(),size=size)
            .databunch(bs=bs).normalize(imagenet_stats))
              
    logging.info("returning the databunch built from df.")
    return(data)


def validate_images(imgpath,classes):

    '''
    Call fastai function verify_images for the each of the classes passed. verify_images will check the if image is valid or not. if not delete it.

    Parameters
    ----------
    imgpath - Path of the image
    classes - The vegetable class name which inturn is the folder name where vegetable images are stored respectively
    
    Returns
    ----------
    None
    
    '''

    logging.info("Validation of images starting .")
    for c in classes:
        print(c)
        verify_images(imgpath/c, delete=True, max_size=500)
    logging.info("Validation of images complete.")