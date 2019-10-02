# -*- coding: utf-8 -*-
#This is  inspired from a guest post by Adrian Rosebrock. Adrian is the author of PyImageSearch.com
import sys
import os
from flask import request, Flask, jsonify
from keras import backend as K
K.clear_session()
print('Clear Keras session before launching it again')
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from PIL import Image
import numpy as np
import io
import pyspark
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql import SQLContext
from pyspark.mllib.classification import SVMWithSGD, SVMModel
#from pyspark.ml.classification import LogisticRegression
import findspark
import tensorflow as tf

findspark.init('C:\spark\spark-2.4.3-bin-hadoop2.7')
sc = SparkContext('local')
spark = SparkSession(sc)
sqlContext = SQLContext(sc)
#Svm_model = os.path.join(os.getcwd())+"\\modelSvm"
#image_path=("C:/Users/Eric/OpenClassroom/P2_classif_appli/images/test/wheaten_terrier/wheaten_terrier_132.jpg")

# API definition
#initialize our Flask application, the Keras model for feature extraction and the SVM trained mode for predcition
app = Flask(__name__)
Keras_model = None
Svm_model= None
Svm_model_path = os.path.join(os.getcwd())+"\\modelSvm"
graph = None
def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global Keras_model
    base_model = VGG16(weights='imagenet')
    global graph
    graph = tf.get_default_graph()
    # Model will produce the output of the 'fc2'layer which is the penultimate neural network layer
    Keras_model = Model(inputs =base_model.input, outputs =base_model.get_layer('fc2').output)
    global Svm_model
    Svm_model = SVMModel.load(sc, Svm_model_path)

def extract_features(image, target):
    #img = image.load_img(image, target_size=(224, 224))
    if image.mode != "RGB":
        image = image.convert("RGB")
    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    #image = imagenet_utils.preprocess_input(image)
    # return the processed image
    #return image
    #x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    image = preprocess_input(image)
    with graph.as_default():
        features = Keras_model.predict(image)#`model._make_predict_function()
    return features.tolist()[0]

@app.route('/predict', methods=['POST'])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        #data["prediction"] = []
        if request.files.get("image"):
            data["prediction"] = []
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            new_image = extract_features(image, target=(224, 224))
            out=Svm_model.predict(new_image)
            #data["prediction"] = str(out)
            if out == 0:
                data["prediction"] = "wheaten_terrier"
            else:
                data["prediction"] = "yorkshire_terrier"
            # indicate that the request was a success
            data["success"] = True    
         
        '''
            # read the image in PIL format
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))
            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []
                    
            # loop over the results and add them to the list of
            # returned predictions.
            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)
            # indicate that the request was a success
            data["success"] = True
            # return the data dictionary as a JSON response
        return jsonify(data)
        '''
    return jsonify(data)
    '''
    #Svm_model = SVMModel.load(sc, Svm_model)
    #new_image = extract_features(keras_model, image_path)
    out=Svm_model.predict(new_image)
    if out==0:
        return "wheaten_terrier"
    else:
        return "yorkshire_terrier"   
    '''
#if this is the main thread of execution first load the model and
#then start the server
if __name__ == "__main__":
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 23456 # If no port provided the port will be set to 23456

    print(("* Loading VGG16 model from Keras, the SVM model and Flask starting server..."
    "please wait until server has fully started"))
    load_model()
    print(" models loaded!")
    #Keras_model.summary()
    #Svm_model.summary()
    app.run(port=port,debug = True)

