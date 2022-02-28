import csv
from flask import Flask
import pandas as pd
import cv2
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

"""
1) Importing dependencies
2) Fetching data from openml
3) Training the model using train_test_split
4) Scaling the data.
5) Fitting the data in the Logistic Regression.
6) make a prediction model.

"""

#fetch data from openml.

X,y = fetch_openml('mnist_784',version=1,return_X_y=True)

#traning the model.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)

#scaling the data.
X_train_scale = X_train/255.0
X_test_scale = X_test/255.0

#Fitting the data in Logistic Regerssion

clf= LogisticRegression(solver="saga",multi_class="multinomial").fit(X_train_scale,y_train)

#Function to predict an image(digits).

def get_prediction(image):
    im_pil= Image.open(image)
    #changing the image's look like in old school movies.
    image_bw= im_pil.convert('L')
    #resizing the old schooler.
    image_bw_resize= image_bw.resize((28,28),Image.ANTIALIAS)
    pixel_filter= 20
    min_pixel= np.percentile(image_bw_resize,pixel_filter)
    image_bw_resize_inverted_scaled= np.clip(image_bw_resize-min_pixel,0,255)
    max_pixel= np.max(image_bw_resize)
    image_bw_resize_inverted_scaled=np.asarray(image_bw_resize_inverted_scaled)/max_pixel
    test_sample= np.array(image_bw_resize_inverted_scaled).reshape(1,784)
    test_pred= clf.predit(test_sample)
    return test_pred[0]