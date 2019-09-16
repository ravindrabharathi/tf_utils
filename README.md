# tf_utils
Utility package for training CNN using tensorflow2, TFRecords, tf.data
Supports the following
1. Download dataset (cifar10 for now..more will be added) from source url
2. Store the Dataset as TFRecords
3. Retreve image data as tf.data.TFRecordDataset
4. Image Augmentation (random-pad_crop, flip-left_right, cutout) of Image Batches
5. Plot images from Dataset 
6. Plot misclassified images 

You may read the instructions below or use the [test notebook](https://github.com/ravindrabharathi/tf_utils/blob/master/test/tf_utils_test.ipynb) to try out the various steps 

### Installation
```
!pip install --upgrade git+https://github.com/ravindrabharathi/tf_utils
```
 

### import data module
```
 import tf_utils.data as ds
```

### set batch size , download data and create tfrecords

```
 ds.batch_size=128

 ds.get_cifar10_and_create_tfrecords()
 ```

### create datasets for training 
```

train_ds=ds.get_train_ds()

test_ds=ds.get_eval_ds()
```
### Image Augmentation
The data module calls the following augmentation for training dataset image batches by default 

```
 cutout(flip_left_right(random_pad_crop(image_batch)))
 
``` 
  
### training
```
#build model 
model=build_model()
#compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy']
              )
#train model 

import numpy as np
batch_size=128
training_steps=np.ceil(50000/batch_size)
test_steps=np.ceil(10000/batch_size)
model.fit(train_ds,epochs=25, steps_per_epoch=training_steps, 
          validation_data=test_ds, validation_steps=test_steps,
          verbose=1)
          
```
### evaluate model 
```

score = model.evaluate(test_ds, steps =test_steps, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

### Visualization 

#### import visualization module
```
import tf_utils.visualize as vz
```
### plot images from training data
```
vz.plot_cifar10_files(train_ds)
```

### plot misclassified images
```
res=vz.get_misclassified_images(model,test_ds)
vz.plot_misclassified_images(res[0],res[1],res[2],res[3],52)
```

#### Note : 
In order to use tensorflow2 on colab , you may use the following code to select tf2 on colab
```
from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass

import tensorflow as tf

```


