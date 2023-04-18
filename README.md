## Bird Classifier
This project leverages Convolutional Neural Networks (CNNs) with keras and tensorflow
to detect species of birds in images.

This is based on the kaggle dataset with 515 bird species: https://www.kaggle.com/datasets/gpiosenka/100-bird-species

## Requirements
1. Python 3.6+ installed
2. Latest version of pip installed

## Setup
1. Download dataset and add to project folder
2. Run `pip install -r requirements.txt` to install dependencies
3. [Optional for training model] Run `python Bird_Classifier.py` to train model
4. To load the flask app, run `python Flask_App.py`
5. Go to localhost:5000 in your browser to view the app

On the web app, users can upload an image of a bird, select a model to evaluate it on, and 
see the top 5 predictions for the bird species.