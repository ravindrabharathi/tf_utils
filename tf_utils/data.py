import tarfile
import os
import sys
from urllib.request import urlretrieve
import numpy as np
import requests

from tqdm import tqdm_notebook as tqdm

import tarfile
from six.moves import cPickle as pickle
from six.moves import xrange

import numpy as np
import pandas as pd

import time
import functools
import psutil

from tf_utils.utils import *
from tf_utils.transform import *

from sklearn.model_selection import train_test_split
from PIL import Image



num_classes=4
batch_size=128
class_names = ['0','1','2','3']
band_list=['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11']


#import tf if not defined 
try:
  tf
except NameError:
    import tensorflow as tf
else:
    pass


def add_random_samples(sample_size=0.25):
  
  return 

# features of a single record , we could add species names too but for now keep it to image and label
rec_features = {
    'B1': tf.io.FixedLenFeature([], tf.string),
    'B2': tf.io.FixedLenFeature([], tf.string),
    'B3': tf.io.FixedLenFeature([], tf.string),
    'B4': tf.io.FixedLenFeature([], tf.string),
    'B5': tf.io.FixedLenFeature([], tf.string),
    'B6': tf.io.FixedLenFeature([], tf.string),
    'B7': tf.io.FixedLenFeature([], tf.string),
    'B8': tf.io.FixedLenFeature([], tf.string),
    'B9': tf.io.FixedLenFeature([], tf.string),
    'B10': tf.io.FixedLenFeature([], tf.string),
    'B11': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
}


###
#FUNCTIONS TO READ TFRECORDS AND CREATE DATASETS  
###   

# function to parse a single record in tfrecords
def _parse_record_function(im_example):
    return tf.io.parse_single_example(im_example, rec_features)

def parse_band(band_info):
  band = tf.decode_raw(band_info, tf.uint8)
  return tf.reshape(band[:4225], shape=(65, 65, 1))

#function to parse a single record and prepare an image/label set for training / evaluation
def parse_record(im_example,distort,distort_fn):
    record = tf.io.parse_single_example(im_example, rec_features)

    bands=[parse_band(record[band]) for band in band_list]
    
    image = tf.concat(bands, -1)
    
    image = tf.cast(image,tf.float32)
    #augment image if needed
    if distort and (distort_fn != None):
      print('distorting...')
    
      image = distort_fn(image)
    
    label = tf.cast(record['label'], tf.int32)
    label = tf.one_hot(label, num_classes)
    
    return image, label
  
  
  
#function to create train and eval datasets 
@timer
def create_train_eval_datasets(train_list,val_list,unlabelled_list=None):
    train_records = tf.data.TFRecordDataset(train_list)
    eval_records = tf.data.TFRecordDataset(val_list)
    
    train_dataset = train_records.map(_parse_record_function)
    eval_dataset = eval_records.map(_parse_record_function)
    #if unlabelled records are specified , create and return unlabelled along with train and val ds
    if unlabelled_list !=None:
      unlabelled_records = tf.data.TFRecordDataset(unlabelled_list)
      unlabelled_dataset = unlabelled_records.map(_parse_record_function)
      return train_dataset, eval_dataset, unlabelled_dataset


    return train_dataset, eval_dataset

#function to create dataset from tfrecords
@timer
def get_tf_dataset_2(recordsfile, batch_size, shuffle=False,distort=False, distort_fn=None):
  #create dataset from tfrecords file  
  files=tf.data.Dataset.list_files(recordsfile)
  dataset = files.interleave(tf.data.TFRecordDataset,cycle_length=4)
  #shuffle 
  if shuffle:
    dataset = dataset.shuffle(buffer_size=10*batch_size)
  #repeat
  dataset = dataset.repeat()
  
  #parse the records - map to parse function
  #setting num_parallel_calls to a value much greater than the number of available CPUs 
  #can lead to inefficient scheduling, resulting in a slowdown
  dataset = dataset.map(map_func=lambda record: parse_record(record, distort, distort_fn),num_parallel_calls=4)


  #although the following link says batch before map is recommended for speed, some tf2 image functions don't seem to handle batches optimally
  #also many tf2 references do map before batch 
  #refer https://stackoverflow.com/questions/50781373/using-feed-dict-is-more-than-5x-faster-than-using-dataset-api
  #so we map and then batch here 
  # batch the records
  dataset = dataset.batch(batch_size=batch_size)  
  #prefetch elements from the input dataset ahead of the time they are requested
  dataset = dataset.prefetch(buffer_size=1)
  
  return dataset 

#create dataset and return an iterator for dataset 
@timer
def get_tf_dataset_in_batches(record_files, batch_size=128, shuffle=False,distort=False,distort_fn=None):
   
  #create dataset 
  dataset = get_tf_dataset_2(record_files, batch_size,shuffle,distort,distort_fn)

  return dataset

#create train data 
@timer
def get_train_ds(train_list,batch_size=batch_size,shuffle=True,distort=True,distort_fn=aug1):
    train_ds = get_tf_dataset_in_batches(train_list, batch_size, shuffle,distort,distort_fn)
    return train_ds
  
  
#create val data
@timer
def get_unlabelled_ds(unlabelled_list,batch_size=batch_size):
    unlabelled_ds = get_tf_dataset_in_batches(unlabelled_list, batch_size)
    return unlabelled_ds

#create test data
@timer
def get_test_ds(test_list,batch_size=batch_size):
    test_ds = get_tf_dataset_in_batches(test_list, batch_size)
    return test_ds     


#get least_confidence_indices  
def get_least_confidence_samples(model,ds,num_steps,total_size,sample_size):
  if sample_size<=1:
    sample_size=int(total_size*sample_size)
  pred=model.predict(ds,steps=num_steps,verbose=0)
  pred=pred[:total_size]
  conf=[]
  indices=[]
  for idx,predxn in enumerate(pred):
    conf.append(np.max(predxn))
    indices.append(idx)
  conf=np.asarray(conf)
  indices=np.asarray(indices)
  least_conf_indices=np.argsort(conf)[:sample_size]
  
  return least_conf_indices

#convert any float values in Label column to int
def convert_to_int(x):
  return int(x)

#add sample with least confident predictions to training dataset and change unlabelled ds accordingly
def add_least_confidence_sample(model,ds,num_steps,total_size,sample_size):
  global train_df, train_df1
  if sample_size<=1:
    sample_size=int(total_size*sample_size)
  pred=model.predict(ds,steps=num_steps,verbose=0)
  pred=pred[:total_size]
  conf=[]
  indices=[]
  for idx,predxn in enumerate(pred):
    conf.append(np.max(predxn))
    indices.append(idx)
  conf=np.asarray(conf)
  indices=np.asarray(indices)
  least_conf_indices=np.argsort(conf)[:sample_size]
  train_df2=train_df.loc[least_conf_indices,:]
  train_df=train_df[~train_df.isin(train_df2)].dropna()
  train_df1=pd.concat([train_df1, train_df2], axis=0)
  train_df=train_df.reset_index(drop=True)
  train_df['Label']=train_df.Label.map(convert_to_int)
  train_df1=train_df1.reset_index(drop=True)
  return train_df1, train_df


#get top-2-confidence-margin indices  
def get_top2_confidence_margin_samples(model,ds,num_steps,total_size,sample_size):
  if sample_size<=1:
    sample_size=int(total_size*sample_size)
  pred=model.predict(ds,steps=num_steps,verbose=0)
  pred=pred[:total_size]
  margins=[]
  indices=[]
  for idx,predxn in enumerate(pred):
    predxn[::-1].sort()
    margins.append(predxn[0]-predxn[1])
    indices.append(idx)
  margins=np.asarray(margins)
  indices=np.asarray(indices)
  least_margin_indices=np.argsort(margins)[:sample_size]
  
  return least_margin_indices

#add top-2-confidence-margin samples for training  
def add_top2_confidence_margin_samples(model,ds,num_steps,total_size,sample_size):
  global train_df, train_df1
  if sample_size<=1:
    sample_size=int(total_size*sample_size)
  pred=model.predict(ds,steps=num_steps,verbose=0)
  pred=pred[:total_size]
  margins=[]
  indices=[]
  for idx,predxn in enumerate(pred):
    predxn[::-1].sort()
    margins.append(predxn[0]-predxn[1])
    indices.append(idx)
  margins=np.asarray(margins)
  indices=np.asarray(indices)
  least_margin_indices=np.argsort(margins)[:sample_size]
  train_df2=train_df.loc[least_margin_indices,:]
  train_df=train_df[~train_df.isin(train_df2)].dropna()
  train_df1=pd.concat([train_df1, train_df2], axis=0)
  train_df=train_df.reset_index(drop=True)
  train_df['Label']=train_df.Label.map(convert_to_int)
  train_df1=train_df1.reset_index(drop=True)
  return train_df1, train_df


#add entropy based sampling for training  
def add_max_entropy_based_samples(model,ds,num_steps,total_size,sample_size):
  global train_df, train_df1
  if sample_size<=1:
    sample_size=int(total_size*sample_size)
  pred=model.predict(ds,steps=num_steps,verbose=0)
  pred=pred[:total_size]
  entropies=[]
  indices=[]
  for idx,predxn in enumerate(pred):
    log2p=np.log2(predxn)
    pxlog2p=predxn * log2p
    n=len(predxn)
    entropy=-np.sum(pxlog2p)/np.log2(n)
    entropies.append(entropy)
    indices.append(idx)
  entropies=np.asarray(entropies)
  indices=np.asarray(indices)
  max_entropy_indices=np.argsort(entropies)[-sample_size:]
  train_df2=train_df.loc[max_entropy_indices,:]
  train_df=train_df[~train_df.isin(train_df2)].dropna()
  train_df1=pd.concat([train_df1, train_df2], axis=0)
  train_df=train_df.reset_index(drop=True)
  train_df['Label']=train_df.Label.map(convert_to_int)
  train_df1=train_df1.reset_index(drop=True)
  return train_df1, train_df
  
