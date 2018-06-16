# Yelp-Restaurant-Photo-Classification
This project is to automatically tag restaurants with nine labels from a bag of images uploaded by users. We used transfer learning with pre-trained CNN models to obtain the features of each image, and followed by a neural network classifier. We achieved a F1 score of 0.743 by using VGG16 feature extractor and LSTM-based neural network classifier, which is a 0.31 increase from a random guesser baseline. 

## Requirements
* install pytorch, keras, python3,cuda,opencv, and sklearn

## Code organization


* `Train_model.ipynb` Implementing feature extraction discribed in section 2.1 of our report and save feature vectors. It also trains our * model discribed in section 2.2 of our report and generate training progress images ploted in section 5.1.1 of our report.
* `ViewPhotoLabels.ipynb` Use our pretrained model to label individual images and show images ploted in section 5.2.1 of our report
* Our training data are stored in the /datasets/ee285s-public/yelpRestaurants/ directory of the server which should be clone to the code    * directory's folder /yelpRestaurants/
* Our pretrained model is uploaded to the dropbox, link:https://www.dropbox.com/s/cl1enggjwc2p185/saved_model_1.pt?dl=0. It should be       put in the code directory

## Authors
Jiawei Li, Yitian Tong, Zhiling Liu, Shiyi Wang
