#!/usr/bin/env python
# coding: utf-8
import os
import tarfile 
import re 
import random
import shutil


def split_on_rawdata(cutoff = None):
    ''' 'Yes' if 1st time extracting and loading features
    '''
    tarf = tarfile.open("images.tar.gz")
    filenamesIntarf = tarf.getnames()
    
    image_list = []
    for img in filenamesIntarf[1:]:
        img = img.partition('/')[2]
        image_list.append(img)
        
    image_path = os.getcwd() + "\\images\\"
    
    if not os.path.exists(image_path):
        tarf.extractall()
    tarf.close()
    
    label = []
    for img in image_list:
        image_name = re.split('\d',img)[0].rstrip('_')
        if image_name not in label:
            label.append(image_name)
        else:
            pass

    train_folder, test_folder = [image_path + str(i) for i in ['train','test']]
    for f in train_folder, test_folder:
        for lb in label:
            subf = str('/' + lb)
            if not os.path.exists(f+subf):
                os.makedirs(f+subf)
    
    image_gr = []
    for lb in label:
        cl = []
        image_gr.append(cl)
        for img in image_list:
            if re.split('\d',img)[0].rstrip('_') == lb:
                cl.append(img)
            else:
                pass  
            
    if cutoff ==  None:
        split_cutoff = float(0.8)
    else:
        split_cutoff = cutoff    
    try:
        for gr in image_gr:
            for i in range(len(gr)):
                class_name = re.split('\d',gr[i])[0].rstrip('_')
                class_name ="\\" + class_name + "\\"
                train_images, test_images = os.listdir(train_folder + class_name), os.listdir(test_folder + class_name)
                if (len(os.listdir(image_path)) > 2):
                    if i < split_cutoff*len(gr): # for each class, take tradeoff*number of examples of this class
                        if not os.path.isfile(train_folder + class_name + gr[i]):
                            os.rename(image_path + "\\" + gr[i], train_folder + class_name + gr[i])
                    else:
                        if not os.path.isfile(test_folder + class_name + gr[i]):
                            os.rename(image_path + '\\' + gr[i], test_folder + class_name + gr[i])    
                else:
                    if len(train_images) < int(split_cutoff*len(gr)):
                        l = random.sample(test_images,int(split_cutoff*len(gr)) - len(train_images))
                        for f in l:
                            if not os.path.isfile(train_folder + class_name + f):
                                os.rename(test_folder + class_name + f, train_folder + class_name + f)
                    elif len(train_images) > int(split_cutoff*len(gr)):
                        l = random.sample(train_images, len(train_images) - int(split_cutoff*len(gr)))
                        for f in l:
                            if not os.path.isfile(test_folder + class_name + f):
                                os.rename(train_folder + class_name + f, test_folder + class_name + f)
                    else:
                        pass  
                if i == 1:
                    print(class_name, len(train_images), len(test_images))
                    print(class_name, len(train_images), len(test_images))
                else:
                    pass  
    except:
        print("no image to move, please load data!") 

feature_extraction_train = os.getcwd() +'\\features\\train'
feature_extraction_test = os.getcwd() +'\\features\\test'
feature_extraction_train1vsAll = os.getcwd() +'\\features\\train1vsAll'
feature_extraction_test1vsAll = os.getcwd() +'\\features\\test1vsAll'

def split(feature_extraction_train = None,feature_extraction_test = None,cutoff = None):
    if cutoff ==  None:
        split_cutoff = float(0.8)
    else:
        split_cutoff = cutoff
    
    for folder in os.listdir(feature_extraction_train):
        feature_extraction_train_folder = os.path.join(feature_extraction_train,folder)
        feature_extraction_test_folder = os.path.join(feature_extraction_test,folder)
        train_images = os.listdir(feature_extraction_train_folder)
        test_images = os.listdir(feature_extraction_test_folder)
        gr = train_images + test_images
        #D[folder] = (len(train_images), len(test_images))
        if len(train_images) < int(split_cutoff*len(gr)):
            l = random.sample(test_images,int(split_cutoff*len(gr)) - len(train_images))
            for f in l:
                if not os.path.isfile(feature_extraction_train_folder + "\\"+ f):
                    os.rename(feature_extraction_test_folder + "\\"+ f, feature_extraction_train_folder + "\\"+ f)
                    
        else :
            l = random.sample(train_images, len(train_images) - int(split_cutoff*len(gr)))
            for f in l:
                if not os.path.isfile(feature_extraction_test_folder + "\\"+ f):
                    os.rename(feature_extraction_train_folder + "\\"+ f, feature_extraction_test_folder + "\\"+ f)
    D = {}
    for folder in os.listdir(feature_extraction_train):
        train_images = os.listdir(feature_extraction_train_folder)
        test_images = os.listdir(feature_extraction_test_folder)
        D[folder] = (len(train_images), len(test_images)) 
       
    return(D)             

def put_to_aws():#(feature_extraction_train,feature_extraction_test,feature_extraction_train1vsAll,feature_extraction_test1vsAll):
    for folder in os.listdir(feature_extraction_train):
        feature_extraction_train_folder = os.path.join(feature_extraction_train,folder)
        feature_extraction_test_folder = os.path.join(feature_extraction_test,folder)
        train_images = os.listdir(feature_extraction_train_folder)
        test_images = os.listdir(feature_extraction_test_folder)
        for image_name in train_images:
            full_image_name = os.path.join(feature_extraction_train_folder,image_name)
            if os.path.isfile(full_image_name):
                shutil.copy(full_image_name, feature_extraction_train1vsAll )
        for image_name in test_images:
            full_image_name = os.path.join(feature_extraction_test_folder,image_name)
            if os.path.isfile(full_image_name):
                shutil.copy(full_image_name, feature_extraction_test1vsAll)        
    
if __name__ == "__main__":
    print("\nWould you like to split the raw data or features?    Yes  or No")
    split_mode = str(input())
    print("\nPlease select a tradeoff from 0 to 1 to split data into the train/test set (exp. 0.6) : ")
    split_cutoff = float(input())
    if split_mode == "Yes":
        split_on_rawdata(cutoff = split_cutoff )       
    else:
        #print("\nPlease define the directory to train folder and test folder")
        #feature_extraction_train = str(input())
        #feature_extraction_test = str(input())
        print('enter Y if spliting and N to prepare features before putting to AWS s3')
        mode = str(input())
        if mode == 'Y':
            split(cutoff = split_cutoff )
        else:
            put_to_aws()
            print("train size: %d" %len(os.listdir(feature_extraction_train1vsAll)))
            print("test size: %d" %len(os.listdir(feature_extraction_test1vsAll)))




