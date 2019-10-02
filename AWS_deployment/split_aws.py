#!/usr/bin/env python
# coding: utf-8
import random
import boto3
import re
import numpy as np

# Define paths to folders and sub-folders where images have been stored. in S3
my_bucket_name = 'opc-imgdetect'
resource = boto3.resource('s3') # use high-level object-oriented API, for low level using client
my_bucket = resource.Bucket(my_bucket_name) 

json_trainfilenames = list(map(lambda x: x.key, my_bucket.objects.filter(Prefix = 'feature_extraction/train1vsAll')))
json_testfilenames = list(map(lambda x: x.key, my_bucket.objects.filter(Prefix = 'feature_extraction/test1vsAll')))
json_trainfilenames = [str(s) for s in json_trainfilenames[1:]]
json_testfilenames = [str(s) for s in json_testfilenames[1:]]
json_trainfilenames = [re.split('/',s)[-1:] for s in json_trainfilenames] # since the list looks like:  ('json_trainfilenames:', [u'feature_extraction/train1vsAll', u'feature_extraction/train1vsAll/Abyssinian_10.json'])
json_testfilenames  =  [re.split('/',s)[-1:] for s in json_testfilenames]
json_trainfilenames = [''.join(s) for s in json_trainfilenames]
json_testfilenames = [''.join(s) for s in json_testfilenames]
json_trainfilenames = [s for s in json_trainfilenames if s.endswith('json')]
json_testfilenames = [s for s in json_testfilenames if s.endswith('json')]
class_names = [list(np.unique(n.split('_')[0])) for n in json_trainfilenames]
print("\nclass_names:", class_names)

def split(json_trainfilenames,json_testfilenames,cutoff = None):
    if cutoff ==  None:
        split_cutoff = float(0.8)
    else:
        split_cutoff = cutoff
    #gr = json_trainfilenames + json_testfilenames
    print("len train feature:", len(json_trainfilenames))
    print("\nlen test feature:", len(json_testfilenames))
    for cln in class_names:
        json_trainfilenames1 = [n for n in json_trainfilenames if n.startswith(cln)]
        json_testfilenames1 = [n for n in json_testfilenames if n.startswith(cln)]
        gr = json_trainfilenames1 + json_testfilenames1
        if len(json_trainfilenames1) < int(split_cutoff*len(gr)):
            l = random.sample(json_testfilenames1,int(split_cutoff*len(gr)) - len(json_trainfilenames1))
            for f in l:
                copy_source = {'Bucket': my_bucket, 'Key': 'feature_extraction/test1vsAll/' + str(f)}
                resource.meta.client.copy(copy_source, my_bucket, 'otherkey')
                print("\nlen train feature:", len(json_trainfilenames))
                print("\nlen test feature:", len(json_testfilenames))
        else:
            l = random.sample(json_trainfilenames1,int(split_cutoff*len(gr)) - len(json_testfilenames1))
            for f in l:
                copy_source = {'Bucket': my_bucket, 'Key': 'feature_extraction/train1vsAll/' + str(f)}
                resource.meta.client.copy(copy_source, my_bucket, 'feature_extraction/test1vsAll/' + str(f))
                print("\nlen train feature:", len(json_trainfilenames))
                print("\nlen test feature:", len(json_testfilenames))
    
    print("\nlen train feature:", len(json_trainfilenames))
    print("\nlen test feature:", len(json_testfilenames))
    
if __name__ == "__main__":
    #split_cutoff = float(input())
    print("\nPlease define the directory to train folder and test folder")
    split(json_trainfilenames,json_testfilenames,cutoff = 0.7 )       





