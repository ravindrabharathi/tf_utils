# tf_utils
Utility package for training CNN using tensorflow2, TFRecords, tf.data
Supports the following
1. Create tfrecords for deepweeds dataset https://github.com/AlexOlsen/DeepWeeds
2. Save TFRecords for Train, eval and Test sets 
3. Retreve image data as tf.data.TFRecordDataset
4. Image Augmentation (random-pad_crop, flip-left_right, cutout) of Image Batches
5. Plot images from Dataset 
6. Plot misclassified images 
7. Plot Confusion Matrix

### Installation
```
!pip install --upgrade git+https://github.com/ravindrabharathi/tf_utils@active_learning
```
 

### import data module
```
 import tf_utils.data as ds
```

### set batch size , download data and create tfrecords

```
 ds.batch_size=128

 ds.create_tfrecords()
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
img_size=256
  
input_layer=Input(shape=(img_size,img_size,3))
base_model=ResNet50(weights='imagenet',include_top=False,input_tensor=input_layer)

x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.15)(x)
x = Dense(512, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional Resnet50 layers
for layer in base_model.layers:
    layer.trainable = False

opt=SGD(lr=0.01,momentum=0.9,nesterov=True)
# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])
#train model 

import numpy as np
EPOCHS=10
#callback_list=[lr_sched,model_cpt]
model.fit(train_ds, epochs=EPOCHS, 
                        steps_per_epoch=np.ceil(len(train_df1)//batch_size), 
                    validation_data=val_ds,
                    validation_steps=np.ceil(len(val_df)//batch_size), 
                    shuffle=True,verbose=1)
          
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
--to be added---

```

#### Note : 
On Colab tf defaults to tensorflow2 
if not , In order to use tensorflow2 on colab , you may use the following code to select tf2 on colab
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


