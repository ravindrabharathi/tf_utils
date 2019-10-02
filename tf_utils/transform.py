
import tensorflow as tf

###
#TRANSFORM FUNCTIONS
###

def center_pad_crop(image,padding=4):
  image=tf.pad(image,[(0, 0), (padding, padding), (padding, padding), (0, 0)], mode='reflect')
  central_fraction=((tf.shape(image)[1]-2*padding)/tf.shape(image)[1])
  print(central_fraction)
  image=tf.image.central_crop(image,central_fraction)

  return image

def random_pad_crop(image,padding=4):
  shp=tf.shape(image)
  
  image=tf.pad(image,[(0, 0), (padding, padding), (padding, padding), (0, 0)], mode='reflect')
  
  image=tf.image.random_crop(image,size=shp)
  return image

def flip_left_right(image):
  return tf.image.random_flip_left_right(image)

def cutout(img, prob=100, size=8, min_size=5, use_fixed_size=True):
  num_dim=len(img.get_shape().as_list())
  if num_dim==4:
    return tf.cond(tf.random.uniform([], 0, 100) > prob, lambda: img , lambda: get_cutout(img,prob,size,min_size,use_fixed_size))
  else:
    return tf.cond(tf.random.uniform([], 0, 100) > prob, lambda: img , lambda: get_cutout_2(img,prob,size,min_size,use_fixed_size))
  
#for batch of images 
def get_cutout(img,prob=50,size=8,min_size=2,use_fixed_size=True):
  
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
def get_cutout_2(img,prob=50,size=8,min_size=2,use_fixed_size=True):
  
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

def aug1(image):
  return cutout(flip_left_right(random_pad_crop(image)))
