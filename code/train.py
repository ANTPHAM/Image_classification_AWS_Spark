# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:31:34 2019

@author: Eric
"""
import os
import re 
import numpy as np
import json
from split import split

#import pyspark
#from pyspark.ml.linalg import SparseVector, DenseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
#from pyspark.sql import SQLContext
from pyspark.mllib.classification import SVMWithSGD
#from pyspark.ml.classification import LogisticRegression
import findspark
findspark.init('C:\spark\spark-2.4.3-bin-hadoop2.7')

sc = SparkContext('local')
spark = SparkSession(sc)

rootPath = os.getcwd() 
feature_extraction_path = os.path.join(rootPath, "features")
feature_extraction_train, feature_extraction_test = [feature_extraction_path + str(i) for i in ['\\train','\\test']] 


def loadAll(filename):
    '''
    filename = the name of an unique file , for instance'yorkshire_terrier_98.json'
    '''
    class1_train=[]
    class0_train=[]
    class1_test=[]
    class0_test=[]
    try:
        folder = re.split('\d',filename)[0].rstrip('_')
        with open(feature_extraction_train + '\\' + folder + '\\' + filename) as json_data:
            ctr = json.load(json_data) 
            if filename.startswith(class1):
                class1_train.append(ctr)
            else:
                class0_train.append(ctr)
    except:
        folder = re.split('\d',filename)[0].rstrip('_')
        with open(feature_extraction_test + '\\' + folder + '\\' + filename) as json_data:
            ctr = json.load(json_data) 
            if filename.startswith(class1):
                class1_test.append(ctr)
            else:
                class0_test.append(ctr)
    return filename,class1_train, class0_train, class1_test, class0_test

def convertAll():
    json_trainfilenames = []
    json_testfilenames = []
    for folder in os.listdir(feature_extraction_train):
        feature_extraction_train_folder = os.path.join(feature_extraction_train,folder)
        feature_extraction_test_folder = os.path.join(feature_extraction_test,folder)
        trainfiles, testfiles = os.listdir(feature_extraction_train_folder), os.listdir(feature_extraction_test_folder)
        if trainmode.lower() == 'yes':
            json_trainfilenames.append(trainfiles)
            json_testfilenames.append(testfiles)
        else:
            if (folder == class1) or (folder == class0) :
                json_trainfilenames.append(trainfiles)
                json_testfilenames.append(testfiles)
            else:
                pass 
    json_trainfilenames = [''.join(n) for i in range(len(json_trainfilenames)) for n in json_trainfilenames[i]]
    json_testfilenames = [''.join(n) for i in range(len(json_testfilenames)) for n in json_testfilenames[i]]
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
    return class1_train, class0_train, class1_test, class0_test

print('\nPreparing all parameters required to build a model')
print("\nAre you looking to train a model based on method 'One vs All'? ")
print("\nIf so,  please enter 'Yes', otherwise enter 'No' ")
trainmode = str(input())


label = ["".join(str(i)) for i in os.listdir(feature_extraction_train)]
print("\nPlease choose one of the following labels as the class1 : \n" +'\n' + ','.join(label))
print('')
class1 = str(input())



if trainmode.lower() == 'no':
    print('\nSelect the class 0 ')
    print('')
    class0 = str(input())
else:
    pass


print("\nPlease select a tradeoff from 0 to 1 to split data into the train/test set (exp. 0.6) : ")
split_cutoff = float(input())
split(feature_extraction_train,feature_extraction_test,cutoff = split_cutoff)

print("\nTurn input data so Mlib model can work")
inputs = convertAll()
class1_train, class0_train, class1_test, class0_test = inputs[0], inputs[1], inputs[2], inputs[3]

# Create the model
lbtrainvec1 = [LabeledPoint(1,class1_train[i]) for i in range(len(class1_train))]
lbtrainvec0 = [LabeledPoint(0,class0_train[i]) for i in range(len(class0_train))]

lbtestvec1 = [LabeledPoint(1,class1_test[i]) for i in range(len(class1_test))]
lbtestvec0 = [LabeledPoint(0,class0_test[i]) for i in range(len(class0_test))]
    
lbtrainvec = lbtrainvec1 + lbtrainvec0 
lbtestvec = lbtestvec1 + lbtestvec0

model_svm = SVMWithSGD.train(sc.parallelize(lbtrainvec), iterations=100, regParam = 0.1, regType = 'l2')
prediction_svm = model_svm.predict(sc.parallelize(class1_test + class0_test)).collect()
prediction_svm=np.array(prediction_svm)
ytest = np.append(np.ones(len(class1_test)),np.zeros(len(class0_test)))

if trainmode.lower() == 'yes':
    print("\nTrained a SVM model based on method 'OnevsAll'")
    print("\nLabel 1 %s ; Lablel 0 : all other classses" % class1)
    print('\nTrain/Test: {}/{}'.format(split_cutoff, round((1-split_cutoff),2)))
else:
    print("\nTrained a SVM model based on method 'OnevsOne'")
    print("\nLabel 1: {} ; Lablel 0: {}".format(class1,class0))
    print('\nTrain/Test: {}/{}'.format(split_cutoff, round((1-split_cutoff),2)))
    
print ('\nAccuracy : %.2f' % float((np.dot(ytest,prediction_svm.T) + np.dot(1-ytest,1-prediction_svm.T))/float(ytest.size)*100) + '%')
#print(prediction_svm)
print('')
#print(ytest)
sc.stop()