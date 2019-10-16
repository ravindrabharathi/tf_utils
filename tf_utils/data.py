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


## for cifar10 
tar = 'cifar-10-python.tar.gz'
url = 'https://www.cs.toronto.edu/~kriz/' + tar

num_classes=10
batch_size=128
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

##for cifar100 ##

class_names_cifar_100=[]

tar_cifar_100="cifar-100-python.tar.gz"
url_cifar_100="https://www.cs.toronto.edu/~kriz/" +tar_cifar_100

###

#import tf if not defined 
try:
  tf
except NameError:
    import tensorflow as tf
else:
    pass


#for tf2 eager execution is enabled by default 
# for lower versions enable eager execution
#if (int(str(tf.__version__)[:1])<2):
  #tf.compat.v1.enable_eager_execution()

#get the cpu cores on current env 
CPU_CORES=get_cpu_num() #get_cpu_cores()  

# features of a single record
rec_features = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}

###
#FUNCTIONS TO DOWLOAD DATASET AND CREATE TFRECORDS
###

#download a file and write to disk  
@timer
def download_file(url, dst):
    file_size = int(requests.head(url).headers["Content-Length"])

    pbar = tqdm(
        total=file_size, initial=0,
        unit='B', unit_scale=True, desc=url.split('/')[-1])

    req = requests.get(url, stream=True)
    with(open(dst, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=10 * 1024):
            if chunk:
                f.write(chunk)
                pbar.update(10 * 1024)
    pbar.close()
    return file_size

#dowload cifar10 data
@timer
def download_cifar10_files():
    path = './'
    if tar not in os.listdir(path):
        download_file(url, tar)
    else:
        print('dataset archive file exists!')

#download cifar100 data
@timer
def download_cifar100_files():
    path = './'
    if tar_cifar_100 not in os.listdir(path):
        download_file(url_cifar_100, tar_cifar_100)
    else:
        print('dataset archive file exists!')        
        
#extract cifar 10 data files downloaded archive 
@timer
def extract_cifar10_files():
    data = './cifar10_data/'
    if os.path.exists(data + 'cifar-10-batches-py/test_batch'):
        print(data + 'cifar-10-batches-py/', 'is not empty and has test_batch file!')

    else:
        tarfile.open(tar, 'r:gz').extractall(data)
        print('Done')
        
        
#extract cifar 100 data files downloaded archive 
@timer
def extract_cifar100_files():
    data = './cifar100_data/'
    if os.path.exists(data + 'cifar-100-python/test'):
        print(data + 'cifar-100-python/', 'is not empty and has test_batch file!')

    else:
        tarfile.open(tar_cifar_100, 'r:gz').extractall(data)
        print('Done')                
        
        
#get filenames in cifar 10 data and split them as train and eval set 
@timer
def _get_file_names(dataset='cifar10'):
    """Returns the file names expected to exist in the input_dir."""
    file_names = {}
    if dataset=='cifar10':
      file_names['train'] = ['data_batch_%d' % i for i in xrange(1, 6)]
      file_names['eval'] = ['test_batch']
    else:
      file_names['train'] = ['train'] 
      file_names['eval'] = ['test'] 
    return file_names

#read the data files 
@timer
def read_pickle_from_file(filename):
    with tf.io.gfile.GFile(filename, 'rb') as f:
        if sys.version_info >= (3, 0):
            data_dict = pickle.load(f, encoding='bytes')
        else:
            data_dict = pickle.load(f)
    return data_dict


#tfrecord features for label and image in cifar 10 dataset 
#label
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#image
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#convert downloaded data files to tfrecords 
@timer
def convert_to_tfrecord(input_files, output_file,dataset='cifar10'):
    """Converts a file to TFRecords."""
    print('Generating %s' % output_file)
    with tf.io.TFRecordWriter(output_file) as record_writer:
        for input_file in input_files:
            # print(input_file)
            data_dict = read_pickle_from_file(input_file)
            data = data_dict[b'data']
            if dataset=='cifar10':
              labels = data_dict[b'labels']
            else:
              labels=data_dict[b'fine_labels']
            num_entries_in_batch = len(labels)
            # print(num_entries_in_batch)

            for i in range(num_entries_in_batch):
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': _bytes_feature(data[i].tobytes()),
                        'label': _int64_feature(labels[i])
                    }))
                record_writer.write(example.SerializeToString())

#function to check if tfrecords exist , else create them 
@timer
def create_tf_records(data_set='cifar10',data_dir='./cifar10_data', output_dir='./', input_dir='cifar-10-batches-py', overwrite=False):
    file_names = _get_file_names(data_set)

    input_dir = os.path.join(data_dir, input_dir)
    for mode, files in file_names.items():
        input_files = [os.path.join(input_dir, f) for f in files]
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
            convert_to_tfrecord(input_files, output_file,data_set)
            print('Done!')

# putting it all together -- download data and create tfrecords 
@timer
def get_cifar10_and_create_tfrecords():
    download_cifar10_files()
    extract_cifar10_files()
    create_tf_records()

# putting it all together cifar100-- download data and create tfrecords 
@timer
def get_cifar100_and_create_tfrecords():
    global num_classes
    global class_names_cifar_100
    num_classes=100
    download_cifar100_files()
    extract_cifar100_files()
    meta = read_pickle_from_file('./cifar100_data/cifar-100-python/meta')
    class_names_cifar_100=[t.decode('utf8') for t in meta[b'fine_label_names']]
    create_tf_records(data_set='cifar100',data_dir='./cifar100_data', output_dir='./', input_dir='cifar-100-python')      
    
###
#FUNCTIONS TO READ TFRECORDS AND CREATE DATASETS  
###   

# function to parse a single record in tfrecords
def _parse_record_function(im_example):
    return tf.io.parse_single_example(im_example, rec_features)

#parse a batch of records if you batch before map
def parse_batch(batch_of_records):
    records=tf.io.parse_example(batch_of_records,rec_features)
    image = tf.io.decode_raw(records['image'], tf.uint8)
    
    #image.set_shape([batch_size * 32 * 32 * 3]) # refer to https://stackoverflow.com/questions/35451948/clarification-on-tf-tensor-set-shape
    image=tf.transpose(tf.reshape(image,[batch_size,3,32,32]),[0,2,3,1])
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
    
    #image.set_shape([batch_size * 32 * 32 * 3]) # refer to https://stackoverflow.com/questions/35451948/clarification-on-tf-tensor-set-shape
    image=tf.transpose(tf.reshape(image,[batch_size,3,32,32]),[0,2,3,1])
    #cast image as float32 as the model requires it
    image = tf.cast(image,tf.float32)
    
    #image augmenation 
    print('distorting image')
    image=aug1(image)
    

    label = tf.cast(records['label'], tf.int32)
    label = tf.one_hot(label, num_classes)

    return image, label  

    


#function to parse a single record and prepare an image/label set for training / evaluation
def parse_record(im_example,distort,distort_fn):
    record = tf.io.parse_single_example(im_example, rec_features)
    
    image = tf.io.decode_raw(record['image'], tf.uint8)
    image.set_shape([32 * 32 * 3])
    image = tf.transpose(tf.reshape(image, [3, 32, 32]), [1, 2, 0])
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
    train_dataset = train_records.map(_parse_record_function)
    eval_dataset = eval_records.map(_parse_record_function)

    return train_dataset, eval_dataset

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
  dataset = dataset.repeat()
  #batch before map is recommended for speed
  #refer https://stackoverflow.com/questions/50781373/using-feed-dict-is-more-than-5x-faster-than-using-dataset-api
  # batch the records
  dataset = dataset.batch(batch_size=batch_size)
  #parse the records - map to parse function
  #setting num_parallel_calls to a value much greater than the number of available CPUs 
  #can lead to inefficient scheduling, resulting in a slowdown
  if distort:
    #print('distorting...')
    dataset = dataset.map(map_func=parse_batch_distort,num_parallel_calls=CPU_CORES)
  else:
    dataset = dataset.map(map_func=parse_batch,num_parallel_calls=CPU_CORES)
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
  dataset = dataset.map(map_func=lambda record: parse_record(record, distort, distort_fn),num_parallel_calls=CPU_CORES)


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
    recordsfile = './train.tfrecords'
  else:
    recordsfile = './eval.tfrecords'
  #create dataset 
  if distort_fn ==None:
    dataset = get_tf_dataset(recordsfile, batch_size,shuffle,distort)
  else:
    dataset = get_tf_dataset_2(recordsfile, batch_size,shuffle,distort,distort_fn)
  # tf version is 2 return dataset , else return an iterator 
  if (int(str(tf.__version__)[:1])<2):
    #create an iterator for the dataset   
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    return iterator
  else:
    return dataset

#create train data 
@timer
def get_train_ds(batch_size=128,shuffle=True,distort=True,distort_fn=None):
    train_ds = get_tf_dataset_in_batches('train', batch_size, shuffle,distort,distort_fn)
    return train_ds
  
  
#create test data
@timer
def get_eval_ds(batch_size=128):
    test_ds = get_tf_dataset_in_batches('test', batch_size)
    return test_ds
