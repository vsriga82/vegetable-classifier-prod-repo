# Vegetable classifer based on deep learning

This application aims to classifiy vegetables using deep learning model. Currently it is trained for classifying 70 classes.

This repository consist of following python files
1) veg_classify_prep_data.py
    - Contains code to prepare data for training the model.
    - Train the model based on following Arguemets are
        * -p or --path  : Path where training images are available
        * -e or --epoch : Number of epochs to be used for training
        * -lr or --maxlr : Learning rate
        * -ml or --model : Model to be used for training (resnet18, resnet34 etc.)
2) veg_classify_predict.py
    - Contains code for single or batch predictions 
    - Prediction is based on following arguements
        * -p or --prediction : Is it single or batch prediction
        * -ml or --modelpath : Path where model pickle file is stored
        * -i or --img : Applicable only for single prediction. The path where the image file is located (e.g image/img1.jpg).
        * -t or --testpath : Applicable only for batch prediction. The folder path where the test images are stored. 
3) build_data.py
    - Contains functions that support building data for training
4) custome_metrics.py
    - Contains functions that support creation of custom metrics for validation
5) test.py
    - Pytest test cases for unit testing.

Start execution with 'veg_classify_prep_data' followed by 'veg_classify_predict'