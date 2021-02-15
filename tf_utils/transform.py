
import tensorflow as tf
#import tensorflow_addons as tfa
import math
import random

###
#TRANSFORM FUNCTIONS
###

#imagenet mean and std
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def center_pad_crop(image,padding=28):
  image=tf.pad(image,[(0, 0), (padding, padding), (padding, padding), (0, 0)], mode='reflect')
  central_fraction=((tf.shape(image)[1]-2*padding)/tf.shape(image)[1])
  print(central_fraction)
  image=tf.image.central_crop(image,central_fraction)

  return image

def pad_img(image,padding=28):
  shp=tf.shape(image)
  num_dim=len(image.get_shape().as_list())
  if num_dim==4:
    image=tf.pad(image,[(0, 0), (padding, padding), (padding, padding), (0, 0)], mode='reflect')
  else:
    image=tf.pad(image,[ (padding, padding), (padding, padding), (0, 0)], mode='reflect')
  
  return image



def random_crop(image,out_shape=(224,224,3)):
  shp=tf.shape(image)
  num_dim=len(image.get_shape().as_list())
  if num_dim==4:
    out_shape=(shp[0],224,224,3)
  return tf.image.random_crop(image,size=out_shape)
  

def random_pad_crop(image,padding=28):
  shp=tf.shape(image)
  num_dim=len(image.get_shape().as_list())
  if num_dim==4:
    image=tf.pad(image,[(0, 0), (padding, padding), (padding, padding), (0, 0)], mode='reflect')
  else:
    image=tf.pad(image,[ (padding, padding), (padding, padding), (0, 0)], mode='reflect')
    
  
  image=tf.image.random_crop(image,size=shp)
  return image

def flip_left_right(image):
  return tf.image.random_flip_left_right(image)

def cutout(img, prob=100, size=56, min_size=14, use_fixed_size=True):
  num_dim=len(img.get_shape().as_list())
  if num_dim==4:
    return tf.cond(tf.random.uniform([], 0, 100) > prob, lambda: img , lambda: get_cutout(img,prob,size,min_size,use_fixed_size))
  else:
    return tf.cond(tf.random.uniform([], 0, 100) > prob, lambda: img , lambda: get_cutout_2(img,prob,size,min_size,use_fixed_size))
  
#for batch of images 
def get_cutout(img,prob=50,size=56,min_size=14,use_fixed_size=True):
  
  shp=tf.shape(img)
  
  
  height = width = tf.shape(img)[1]
  
  channel = tf.shape(img)[-1]
  
  

  #get cutout size and offsets 
  if (use_fixed_size==True):
    s=size
  else:  
    s=tf.random.uniform([], min_size, size, tf.int32) # use a cutout size between 5 and size 

  x1 = tf.random.uniform([], 0, height+1-s , tf.int32) # get the x offset from top left
  y1 = tf.random.uniform([], 0, width+1-s , tf.int32) # get the y offset from top left 
  
  # create the cutout slice and the mask 
  img1 = tf.ones_like(img)  
  #print(tf.shape(img1))
  cut_slice = tf.slice(
  img1,
  [0,x1, y1, 0],
  [shp[0],s, s, channel]
     )
  
  
  #create mask similar in shape to input image with cutout area having ones and rest of the area padded with zeros 
  mask = tf.image.pad_to_bounding_box(
    cut_slice,
    x1,
    y1,
    height,
    width
  )
  
  
  #invert the zeros and ones 
  mask = tf.ones_like(mask ) - mask
  
  #print(tf.shape(mask))
  
  tmp_img = tf.multiply(img,mask)
  
  cut_img =tmp_img
   
  return cut_img 

#for single image
def get_cutout_2(img,prob=50,size=56,min_size=14,use_fixed_size=True):
  
  shp=tf.shape(img)
  
  
  height = width = tf.shape(img)[1]
  
  channel = tf.shape(img)[-1]
  
  

  #get cutout size and offsets 
  if (use_fixed_size==True):
    s=size
  else:  
    s=tf.random.uniform([], min_size, size, tf.int32) # use a cutout size between 5 and size 

  x1 = tf.random.uniform([], 0, height+1-s , tf.int32) # get the x offset from top left
  y1 = tf.random.uniform([], 0, width+1-s , tf.int32) # get the y offset from top left 
  
  # create the cutout slice and the mask 
  img1 = tf.ones_like(img)  
  #print(tf.shape(img1))
  cut_slice = tf.slice(
  img1,
  [x1, y1, 0],
  [s, s, channel]
     )
  
  
  #create mask similar in shape to input image with cutout area having ones and rest of the area padded with zeros 
  mask = tf.image.pad_to_bounding_box(
    cut_slice,
    x1,
    y1,
    height,
    width
  )
  
  
  #invert the zeros and ones 
  mask = tf.ones_like(mask ) - mask
  
  #print(tf.shape(mask))
  
  tmp_img = tf.multiply(img,mask)
  
  cut_img =tmp_img
   
  return cut_img

def resize_img(img,size):
  if (int(str(tf.__version__)[:1])<2):
    tf.image.resize_images(img,size,method=tf.image.ResizeMethod.BICUBIC)
  else:
    return tf.image.resize(img,size,method=tf.image.ResizeMethod.BICUBIC) 

def normalize_img(img,mean=mean,std=std):
  return ((img/255.0)-mean)/std

def normalize_img2(img,mean,std):
  return img/255.0


def aug1(image):
  #return cutout(flip_left_right(random_pad_crop(image)))
  return cutout(flip_left_right(image))
