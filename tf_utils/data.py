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

import time
import functools
import psutil

from tf_utils.utils import *
from tf_utils.transform import *

from sklearn.model_selection import train_test_split
from PIL import Image


num_classes=9
batch_size=128
class_names = ['0','1','2','3','4','5','6','7','8']
global species_names,train_df,train_df1,val_df,test_df

#import tf if not defined 
try:
  tf
except NameError:
    import tensorflow as tf
else:
    pass


def split_train_test_df(data_df):
  global train_df, train_df1,test_df
  train_df, test_df = train_test_split(data_df, test_size=0.2,shuffle=True,random_state=42)
  train_df1,train_df=train_test_split(train_df, test_size=0.75,shuffle=True,random_state=42)
  return train_df1,train_df,test_df

def get_random_train_df():
  global train_df,train_df1
  train_df2,train_df=train_test_split(train_df, test_size=0.75,shuffle=True,random_state=42)
  train_df1=pd.concat([train_df1, train_df2], axis=0).reset_index()
  return train_df1,train_df

def get_train_val_test_df(data_df):
  global train_df1,val_df,test_df
  train_df, test_df = train_test_split(data_df, test_size=0.2,shuffle=True,random_state=42)
  train_df1, val_df = train_test_split(train_df, test_size=0.25,shuffle=True,random_state=42)
  return train_df1,val_df,test_df


def get_class_map_and_species_names(data_df):
  class_map=data_df[['Label','Species']]
  class_map=class_map.drop_duplicates()
  class_map=class_map.sort_values(by=['Label']).reset_index(drop=True)
  species_names=class_map.to_dict()['Species']
  return class_map,species_names



# features of a single record , we could add species names too but for now keep it to image and label
rec_features = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}


#get filenames in image data and split them as train and eval set 

def _get_file_names():
    """Returns the file names expected to exist in the input_dir."""
    file_names = {}
    file_names['train'] = list(zip(train_df1.Filename, train_df1.Label))
    file_names['unlabelled'] = list(zip(train_df.Filename, train_df.Label))
    file_names['test'] = list(zip(test_df.Filename, test_df.Label))
    
    return file_names

#tfrecord features for label and image in cifar 10 dataset 
#label
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#image
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#convert downloaded data files to tfrecords 
@timer
def convert_to_tfrecord(input_files, output_file,images_dir=''):
    """Converts a file to TFRecords."""
    print('Generating %s' % output_file)
    with tf.io.TFRecordWriter(output_file) as record_writer:
        for filename, label in input_files:
            if images_dir !='':
              filename=os.path.join(images_dir,filename)
            img = Image.open(filename)
            img = np.array(img.resize((256,256)))

            feature = { 'label': _int64_feature(label),
                         'image': _bytes_feature(img.tostring()) }

            # Create an example protocol buffer

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            
            
            record_writer.write(example.SerializeToString())

#function to check if tfrecords exist , else create them 
@timer
def create_tf_records(data_dir='./data', output_dir='./tf_train',images_dir='./data', overwrite=False):
    file_names = _get_file_names()
    if os.path.exists(output_dir):
      pass
    else:  
      os.mkdir(output_dir)
    input_dir = data_dir
    for mode, files in file_names.items():
      
      #for file,label in files:
      #input_files =[os.path.join('./',f) for f in input_files]
      output_file = mode + '.tfrecords'
      if ((output_file in os.listdir(output_dir)) & (overwrite == False)):
          print(output_file, 'exists!')
      else:
          output_file = os.path.join(output_dir, output_file)
          try:
              os.remove(output_file)
          except OSError:
              pass
            # Convert to tf.train.Example and write the to TFRecords.
          convert_to_tfrecord(files, output_file,images_dir)
          print('Done!')
        
        



###
#FUNCTIONS TO READ TFRECORDS AND CREATE DATASETS  
###   

# function to parse a single record in tfrecords
def _parse_record_function(im_example):
    return tf.io.parse_single_example(im_example, rec_features)

#parse a batch of records if you batch before map
def parse_batch(batch_of_records):
    print('---parse batch -----')
    records=tf.io.parse_example(batch_of_records,rec_features)
    image = tf.io.decode_raw(records['image'], tf.uint8)
    print(image.dtype,image.shape,len(image))
    #image.set_shape([batch_size * 224 * 224 * 3]) # refer to https://stackoverflow.com/questions/35451948/clarification-on-tf-tensor-set-shape
    #image=tf.transpose(tf.reshape(image,[batch_size,3,224,224]),[0,2,3,1])
    image=tf.reshape(image,[batch_size,256,256,3])
    #cast image as float32 as the model requires it
    image = tf.cast(image,tf.float32)
    #augment image if needed
    #send image to augment fn here 
    #image=aug_fn(image)
    ##

    label = tf.cast(records['label'], tf.int32)
    label = tf.one_hot(label, num_classes)

    return image, label
  
#parse a batch of records if you batch before map
def parse_batch_distort(batch_of_records):
    records=tf.io.parse_example(batch_of_records,rec_features)
    image = tf.io.decode_raw(records['image'], tf.uint8)
    
    #image.set_shape([batch_size * 224 * 224 * 3]) # refer to https://stackoverflow.com/questions/35451948/clarification-on-tf-tensor-set-shape
    #image=tf.transpose(tf.reshape(image,[batch_size,3,224,224]),[0,2,3,1])
    image=tf.reshape(image,[batch_size,256,256,3])
    #cast image as float32 as the model requires it
    image = tf.cast(image,tf.float32)
    
    #image augmenation 
    print('distorting image')
    #image=aug1(image)
    

    label = tf.cast(records['label'], tf.int32)
    label = tf.one_hot(label, num_classes)

    return image, label  

    


#function to parse a single record and prepare an image/label set for training / evaluation
def parse_record(im_example,distort,distort_fn):
    record = tf.io.parse_single_example(im_example, rec_features)
    
    image = tf.io.decode_raw(record['image'], tf.uint8)
    image.set_shape([256 * 256 * 3])
    #image = tf.transpose(tf.reshape(image, [3, 256, 256]), [1, 2, 0])
    image = tf.reshape(image, [256, 256,3])
    #image = tf.reshape(image,[32,32,3]) #check this ..this doesn't seem to give the right image 
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
def create_train_eval_datasets():
    train_records = tf.data.TFRecordDataset('./train.tfrecords')
    eval_records = tf.data.TFRecordDataset('./eval.tfrecords')
    test_records = tf.data.TFRecordDataset('./test.tfrecords')
    train_dataset = train_records.map(_parse_record_function)
    eval_dataset = eval_records.map(_parse_record_function)
    test_dataset = test_records.map(_parse_record_function)

    return train_dataset, eval_dataset, test_dataset

#function to create dataset from tfrecords
@timer
def get_tf_dataset(recordsfile, batch_size, shuffle=False,distort=False):
  #create dataset from tfrecords file  
  files=tf.data.Dataset.list_files(recordsfile)
  dataset = files.interleave(tf.data.TFRecordDataset,cycle_length=4)
  #shuffle 
  if shuffle:
    dataset = dataset.shuffle(buffer_size=10*batch_size)
  #repeat
  #if recordsfile != './tf_train/test.tfrecords':
  dataset = dataset.repeat()
  #batch before map is recommended for speed
  #refer https://stackoverflow.com/questions/50781373/using-feed-dict-is-more-than-5x-faster-than-using-dataset-api
  # batch the records
  dataset = dataset.batch(batch_size=batch_size)
  #parse the records - map to parse function
  #setting num_parallel_calls to a value much greater than the number of available CPUs 
  #can lead to inefficient scheduling, resulting in a slowdown
  if distort:
    print('distorting...')
    dataset = dataset.map(map_func=parse_batch_distort,num_parallel_calls=4)
  else:
    dataset = dataset.map(map_func=parse_batch,num_parallel_calls=4)
  #prefetch elements from the input dataset ahead of the time they are requested
  dataset = dataset.prefetch(buffer_size=1)
  
  return dataset

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
def get_tf_dataset_in_batches(recordstype='train', batch_size=128, shuffle=False,distort=False,distort_fn=None):
  #switch file name based on train or test data
  if recordstype == 'train':
    recordsfile = './tf_train/train.tfrecords'
  elif recordstype== 'unlabelled':
    recordsfile = './tf_train/unlabelled.tfrecords'
  elif recordstype== 'test':
    recordsfile = './tf_train/test.tfrecords'  
  #create dataset 
  if distort_fn ==None:
    dataset = get_tf_dataset(recordsfile, batch_size,shuffle,distort)
  else:
    dataset = get_tf_dataset_2(recordsfile, batch_size,shuffle,distort,distort_fn)
  # tf version is 2 return dataset , else return an iterator 
  if (int(str(tf.__version__)[:1])<2):
    #create an iterator for the dataset  
    print('returning iterator for ',tf.__version__)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    return iterator
  else:
    print('returning dataset for ',tf.__version__)
    return dataset

#create train data 
@timer
def get_train_ds(batch_size=batch_size,shuffle=True,distort=True,distort_fn=aug1):
    train_ds = get_tf_dataset_in_batches('train', batch_size, shuffle,distort,distort_fn)
    return train_ds
  
  
#create val data
@timer
def get_unlabelled_ds(batch_size=batch_size):
    unlabelled_ds = get_tf_dataset_in_batches('unlabelled', batch_size)
    return unlabelled_ds

#create test data
@timer
def get_test_ds(batch_size=batch_size):
    test_ds = get_tf_dataset_in_batches('test', batch_size)
    return test_ds     
