#!/usr/bin/env python
# coding: utf-8
import re 
import numpy as np
import json
import boto3

from pyspark.mllib.regression import LabeledPoint
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.mllib.classification import SVMWithSGD  

sc = SparkContext()
spark = SparkSession(sc)

# Define paths to folders and sub-folders where images have been stored. in S3
my_bucket_name = 'opc-imgdetect'
resource = boto3.resource('s3') # use high-level object-oriented API, for low level using client
my_bucket = resource.Bucket(my_bucket_name) 

# extracting the name of each file and put all into a list
json_trainfilenames = list(map(lambda x: x.key, my_bucket.objects.filter(Prefix = 'feature_extraction/train')))
json_testfilenames = list(map(lambda x: x.key, my_bucket.objects.filter(Prefix = 'feature_extraction/test')))
json_trainfilenames = [str(s) for s in json_trainfilenames[1:]]
json_testfilenames = [str(s) for s in json_testfilenames[1:]]
json_trainfilenames = [re.split('/',s)[-1:] for s in json_trainfilenames] # cause the list looks like this ('json_trainfilenames:', [u'feature_extraction/train1vsAll', u'feature_extraction/train1vsAll/Abyssinian_10.json'])
json_testfilenames  =  [re.split('/',s)[-1:] for s in json_testfilenames]
json_trainfilenames = [''.join(s) for s in json_trainfilenames]
json_testfilenames = [''.join(s) for s in json_testfilenames]
json_trainfilenames = [s for s in json_trainfilenames if s.endswith('json')]
json_testfilenames = [s for s in json_testfilenames if s.endswith('json')]
print(json_testfilenames)

def loadAll(filename):
    class1_train=[]
    class0_train=[]
    class1_test=[]
    class0_test=[]
    my_bucket_name = 'opc-imgdetect'
    resource = boto3.resource('s3') 
    try:
        object = resource.Object(my_bucket_name,'feature_extraction/train/' + str(filename)).get()
        object = object['Body'].read().decode('utf-8')
        object =  str(object)
        features= [i.strip() for i in  object.strip('[]').split(',')]
        features= [float(i) for i in features]
        if filename.startswith(class1):
            class1_train.append(features) 
        else:
            class0_train.append(features) 
    except:
        object1 = resource.Object(my_bucket_name,'feature_extraction/test/' + str(filename)).get()
        object1 = object1['Body'].read().decode('utf-8')
        object1 = str(object1)
        features1= [i.strip() for i in  object1.strip('[]').split(',')]
        features1= [float(i) for i in features1]
        if filename.startswith(class1):
            class1_test.append(features1) 
        else:
            class0_test.append(features1)
    return filename, class1_train, class0_train, class1_test, class0_test

def convertAll(json_trainfilenames,json_testfilenames):
    if trainmode.lower() == 'yes':
        json_trainfilenames = json_trainfilenames
        json_testfilenames = json_testfilenames
    else:
        json_trainfilenames = [n for n in json_trainfilenames if (n.startswith(class1)) or (n.startswith(class0))]
        json_testfilenames = [n for n in json_testfilenames if (n.startswith(class1)) or (n.startswith(class0))]
    print(json_testfilenames)
trainmode = 'yes'
print("\nTurn input data so Mlib model can work")
inputs = convertAll(json_trainfilenames,json_testfilenames)
class1_train, class0_train, class1_test, class0_test = inputs[0], inputs[1], inputs[2], inputs[3]
'''
    rdd_trainfilenames = sc.parallelize(json_trainfilenames)
    rdd_testfilenames = sc.parallelize(json_testfilenames)    
    # we are going to use RDD oject to handle features
    trainRDD = rdd_trainfilenames.map(loadAll)
    testRDD = rdd_testfilenames.map(loadAll)
    train_features = trainRDD.collect()
    test_features = testRDD.collect()
    class1_train = [train_features[i][1][0] for i in range(len(train_features))
                    if ((len(train_features[i][1]) !=0) and (train_features[i][1][0] is not None))]# remove Null and NoneType elements
    class0_train = [train_features[i][2][0] for i in range(len(train_features))
                    if ((len(train_features[i][2]) !=0) and (train_features[i][2][0] is not None))]
    class1_test = [test_features[i][3][0] for i in range(len(test_features))
                   if ((len(test_features[i][3]) !=0) and (test_features[i][3][0] is not None))]
    class0_test = [test_features[i][4][0] for i in range(len(test_features))
                   if ((len(test_features[i][4]) !=0) and (test_features[i][4][0] is not None))]
    return class1_train, class0_train, class1_test, class0_test, json_trainfilenames, json_testfilenames
    
print('\nPreparing all parameters required to build a model')
print("\nAre you looking to train a model based on method 'One vs All'? ")
print("\nIf so,  please enter 'Yes', otherwise enter 'No' ")
trainmode = 'yes'

print("\nTurn input data so Mlib model can work")
inputs = convertAll(json_trainfilenames,json_testfilenames)
class1_train, class0_train, class1_test, class0_test = inputs[0], inputs[1], inputs[2], inputs[3]

print('')
class1 = str('Ragdoll')
print("Target class:", class1)

if trainmode.lower() == 'no':
    print('\nSelect the class 0 ')
    print('')
    class0 = 'shiba_inu'
else:
    pass

# SVMWithSGD ( mlib.Classification)
# In order to train a SVM model, we need first to  handle original inputs , there by these can be explored by the model.
# We will be using the function Labeledpoint to do that.

print("size of the train set class 1: %d" % len(class1_train))
print("size of the train set class 0: %d" % len(class0_train))
print("size of the test set class 1: %d" % len(class1_test))
print("size of the test set class 0: %d" % len(class0_test))
trainsetsize = '%.2f' % float((len(class1_train) + len(class0_train))/float(len(class1_train) + len(class0_train) + len(class1_test) + len(class0_test))*100)
print("size of the training set: %.2f" %float(trainsetsize) + '%')

lbtrainvec1 = [LabeledPoint(1,class1_train[i]) for i in range(len(class1_train))]
lbtrainvec0 = [LabeledPoint(0,class0_train[i]) for i in range(len(class0_train))]
lbtrainvec = lbtrainvec1 + lbtrainvec0
lbtestvec1 = [LabeledPoint(1,class1_test[i]) for i in range(len(class1_test))]
lbtestvec0 = [LabeledPoint(0,class0_test[i]) for i in range(len(class0_test))]
lbtestvec = lbtestvec1 + lbtestvec0

# train and evaluate the model
model_svm = SVMWithSGD.train(sc.parallelize(lbtrainvec), iterations=100, regParam = 0.1, regType = 'l2')
prediction_svm = model_svm.predict(sc.parallelize(class1_test + class0_test)).collect()

s3object = resource.Object(my_bucket_name, "outSvm_1vsA.json") # create a .json file on s# to store the output of the model
s3object.put(Body=(bytes(json.dumps(prediction_svm).encode('UTF-8')))) # write the output on the this file

prediction_svm=np.array(prediction_svm)
ytest = np.append(np.ones(len(class1_test)),np.zeros(len(class0_test)))

acc = '%.2f' %float((np.dot(ytest,prediction_svm.T) + np.dot(1-ytest,1-prediction_svm.T))/float(ytest.size)*100) 
summary = 'Target class: {}  Size of training set: {}  Accuracy of OnevsAll based SVM model: {}'.format( str(class1), str(trainsetsize) +'%', str(acc) + '%')
print(summary)
s3object1 = resource.Object(my_bucket_name, "accuracy_1vsA.txt") # create a .txt file in the s3 bucket to store the summary
s3object1.put(Body=(bytes(json.dumps(summary).encode('UTF-8')))) # write the summary on the above file

# close Spark context
sc.stop()
'''




