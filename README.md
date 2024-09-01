# Garbage-Classification
This project focuses on developing a machine learning model to classify garbage into multiple categories. The aim is to automate the sorting process by identifying the type of waste—such as plastic, glass, paper, metal, or organic material—based on the provided images. The project can serve as a stepping stone toward building more complex waste management systems that contribute to recycling and environmental sustainability.

## Table of Content

- [Import Libraries](#Import_Libraries)
- [Read Dataset](#Read_Dataset)
- [Visualization](#Visualization)
- [Modeling](#Modeling)
- [Confusion_matrix](#Confusion_matrix)
- [Accuracy](#Accuracy)

## Import_Libraries
- import numpy as np
- import pandas as pd
- import os
- import cv2
- import matplotlib.pyplot as plt
- import seaborn as sns
- import tensorflow as tf
- import keras 
- from tqdm import tqdm
- from keras.callbacks import EarlyStopping, ModelCheckpoint
- from sklearn.metrics import confusion_matrix, accuracy_score
- from sklearn.metrics import classification_report
- from sklearn.model_selection import train_test_split
- from sklearn.preprocessing import LabelEncoder
- import glob 
- import pandas as pan
- import matplotlib.pyplot as plotter
- import warnings
- warnings.filterwarnings('ignore')
