# Yelp-Restaurant-Photo-Classification
This project is to automatically tag restaurants with nine labels from a bag of images uploaded by users. We used transfer learning with pre-trained CNN models to obtain the features of each image, and followed by a neural network classifier. We achieved a F1 score of 0.743 by using VGG16 feature extractor and LSTM-based neural network classifier, which is a 0.31 increase from a random guesser baseline. 

## Requirements
* install pytorch, keras, python3,cuda,opencv, and sklearn

## Code organization

* `Model_SVM.ipynb` This is the first model we tried, using PCA+SVM method, we use this as a baseline for our model.
* `Train_model.ipynb` Implementing feature extraction discribed in section 2.1 of our report and save feature vectors. It also trains our * model discribed in section 2.2 of our report and generate training progress images ploted in section 5.1.1 of our report. To test this file, there are several things to be noticed:

 * 1:When extracting the feature, you can choose to run "get_total_feature(train_num,test_num,img_num,vgg)", this method is much flexible but could waste a lot of time, since you need to run that every time you train a model; 

 *train_num represent the number of business_id we want for training;*
 *test_num represent the number of business_id we want for testing;*
 *img_num represent the maximum number of images we want to get from each business;*
 *vgg is the pretrain model we choose*

 * 2: Another way we provide is as what we specified in section 3, we use save_data(batch,img_num,vgg) to save all the feature in a file,then you can use load_data(train_num,test_num) to load feature every time you train a new model, however this could took a long time to run the first time, here train_num,test_num means the number of files we want.

 *batch represent the number of business we want in a file, for example, if batch = 100, there are total 2000 business, therefore it will produce 20 files with features after running.*

 * 3:model_name is the name of the model we saved, please keep that fixed with the model_name we loaded in 'ViewPhotoLabels.ipynb'

* `ViewPhotoLabels.ipynb` Use our pretrained model to label individual images and show images ploted in section 5.2.1 of our report
* Our training data are stored in the /datasets/ee285s-public/yelpRestaurants/ directory of the server which should be clone to the code    * directory's folder /yelpRestaurants/
* Our pretrained model is uploaded to the dropbox, link:https://www.dropbox.com/s/cl1enggjwc2p185/saved_model_1.pt?dl=0. It should be       put in the code directory

## Authors
Jiawei Li, Yitian Tong, Zhiling Liu, Shiyi Wang
