#!/usr/bin/env python
# coding: utf-8
import json
import os
from keras import backend as K
K.clear_session()
print('Clear Keras session before launching it again')
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import tensorflow as tf
import split



print("\nPlease select a tradeoff from 0 to 1 to split data into the train/test set (exp. 0.6) : ")
split_cutoff = float(input())
print("\nfor spliting the image data enter 'Y'; for already extracted features enter = N?")
split_mode = str(input())
if split_mode == "Y":
    split.split_on_rawdata(cutoff = split_cutoff )      
else:
    print("\nYou might split extracted features and train the model")

print("\npre-trained model loading!")    
keras_model = None
graph = None
def load_model():
    global keras_model
    base_model = VGG16(weights='imagenet')
    global graph
    graph = tf.get_default_graph()
    # Model will produce the output of the 'fc2'layer which is the penultimate neural network layer
    keras_model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
load_model()
print("\nPre-trained models loaded successfully!")    

def extract_features(image_path, mode = None):
    # load an image , convert it  into features and store them in a list
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        with graph.as_default():
                features = keras_model.predict(img)#`model._make_predict_function()
                if mode == 'array':
                        return features[0]
                elif mode == 'list':
                        return features.tolist()[0]
                else:
                    return features[0]
    except:
        pass

rootPath = os.getcwd()  
base_dir = os.getcwd()+'\\images'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'test')

label = ["".join(str(i)) for i in os.listdir(train_dir)]
print('\nLabels in the data set: ' + ', '.join(label))

if not os.path.exists(os.getcwd() +'\\features'):
    for f in ['train','test']:
        os.makedirs('features\\%s' %f)
        
feature_extraction_path = os.path.join(rootPath, "features")
feature_extraction_train, feature_extraction_test = [feature_extraction_path + str(i) for i in ['\\train','\\test']] 

def extractAll():
    for folder in os.listdir(train_dir):
        if not os.path.exists(feature_extraction_train +'\\' + folder):
            os.makedirs("features\\train\\%s" %folder)
        for img in os.listdir(train_dir + "\\" + folder):
            img_path = os.path.join(train_dir + "\\" + folder,img)
            imgname = ''.join(img.split('.')[:-1]) # remove .jpeg
            features = extract_features(img_path, mode ='list')
            filepath = feature_extraction_train + "\\" + folder + '\\'+ imgname + ".json"
            if not os.path.isfile(filepath):
                with open(filepath, 'w') as out:
                    json.dump(features, out)        
    
    for folder in os.listdir(validation_dir):
        if not os.path.exists(feature_extraction_test +'\\' + folder):
            os.makedirs("features\\test\\%s" %folder)
        for img in os.listdir(validation_dir + "\\" + folder):
            img_path = os.path.join(validation_dir + "\\" + folder,img)
            imgname = ''.join(img.split('.')[:-1]) # remove .jpeg
            features = extract_features(img_path, mode = 'list')
            filepath = feature_extraction_test + "\\" + folder +'\\'+ imgname + ".json"
            if not os.path.isfile(filepath):
                with open(filepath, 'w') as out:
                    json.dump(features, out)   
if __name__ == "__main__":
    load_model()
    extractAll()

