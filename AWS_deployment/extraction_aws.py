#!/usr/bin/env python
# coding: utf-8
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
import tensorflow as tf
from PIL import Image
import boto3
from io import BytesIO
import re
import json

# create some global variables  
keras_model = None
graph = None
def load_model():
    global keras_model
    base_model = VGG16(weights='imagenet')
    global graph
    graph = tf.get_default_graph()
    keras_model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
load_model() # in case of suing this script as an imported modlule
print("pre-trained models loaded successfully!")    

def extract_features(image, mode = None):
    #load an real image object (not a path to image)  and convert it to features stored in a list
    try:
        img = Image.open(BytesIO(image)) # using PIL to open the real image
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((224,224))
        img = img_to_array(img)# image.img_to_array is a keras function
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        with graph.as_default():
            features = keras_model.predict(img)
        if mode == 'array':
            return features[0]
        elif mode == 'list':
            return features.tolist()[0]
        else:
            return features[0]    
    except:
        pass   

# Define paths to folders and sub-folders where images have been stored in S3
# Using boto3 library to interact with S3
my_bucket_name = 'opc-imgdetect'
resource = boto3.resource('s3') #high-level object-oriented API
my_bucket = resource.Bucket(my_bucket_name) 

train_path = list(my_bucket.objects.filter(Prefix='images/train')) # pointing to the last element in all subfolders : [s3.ObjectSummary(bucket_name='opc-imgdetect', key=u'images/train/Abyssinian/Abyssinian_10.jpg'),...]
test_path = list(my_bucket.objects.filter(Prefix='images/test'))

def extractAll():
    for train_image_path in train_path:
        feature_train_path = list(my_bucket.objects.filter(Prefix='feature_extraction/train1vsAll'))
        imgjson = str(train_image_path.key) # giving : images/train/yorkshire_terrier/yorkshire_terrier_106.jpg
        imgjson = re.split('/',imgjson)[-1:] # getting out 'yorkshire_terrier_106.jpg 
        imgjson = ''.join(imgjson)
        if imgjson.endswith("jpg"):
            imgjson = re.findall(r'((?<=).*?(?=.jpg))',imgjson)[0] + '.json' 
        else:
            pass
        if imgjson not in feature_train_path:
            object = train_image_path.get()
            data = object['Body'].read()
            features = extract_features(data, mode =  "list")
            s3object = resource.Object(my_bucket_name, 'feature_extraction/train1vsAll/' + imgjson) 
            s3object.put(Body=(bytes(json.dumps(features).encode('UTF-8'))))
        else:
            pass
      
    for test_image_path in test_path:
        feature_test_path = list(my_bucket.objects.filter(Prefix = 'feature_extraction/test1vsAll'))
        imgjson = str(test_image_path.key) 
        imgjson = re.split('/',imgjson)[-1:]  
        imgjson = ''.join(imgjson)
        if imgjson.endswith("jpg"):
            imgjson = re.findall(r'((?<=).*?(?=.jpg))',imgjson)[0] + '.json' 
        else:
            pass
        if imgjson not in feature_test_path:
            object = test_image_path.get()
            data = object['Body'].read()
            features = extract_features(data, mode = "list")
            s3object = resource.Object(my_bucket_name, 'feature_extraction/test1vsAll/' + imgjson) 
            s3object.put(Body=(bytes(json.dumps(features).encode('UTF-8'))))
        else:
            pass

if __name__ == "__main__":
    load_model() #in case of running directly this script
    extractAll()
