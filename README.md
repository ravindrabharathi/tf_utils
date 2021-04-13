# tf_utils
Utility package for using active learning strategies for labelling image datesets and training CNN using tensorflow2, TFRecords, tf.data
Supports the following
1. Create tfrecords for deepweeds dataset https://github.com/AlexOlsen/DeepWeeds
2. Save TFRecords for Train, eval and Test sets 
3. Retreve image data as tf.data.TFRecordDataset
4. Image Augmentation (random-pad_crop, flip-left_right, cutout) of Image Batches
5. Active Learning data points selection for labelling using Least confidence sampling , Margin sampling (Entropy sampling to be added) 
6. Random sampling method for getting a baseline is supported 
7. Plot images from Dataset 
8. Plot Confusion Matrix

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
### Plot images from Dataset 

```
vz.plot_images_from_ds(train_ds,"Samples from Train Dataset",species_names)
```
![image](https://user-images.githubusercontent.com/597097/114504802-11cd1580-9c4d-11eb-9d60-c7835d705396.png)

  
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
### Sampling methods 
```
num_steps=np.ceil(len(train_df)/batch_size)
total_samples=len(train_df)
sample_size=0.25
```
#### Random Sampling 
```
train_df1,train_df=ds.add_random_samples(sample_size=sample_size)
```

#### Least confidence sampling 
```
train_df1,train_df=ds.add_least_confidence_sample(model,unlabelled_ds,num_steps,total_samples,sample_size)
```

#### Margin Sampling 
```
train_df1,train_df=ds.add_top2_confidence_margin_samples(model,unlabelled_ds,num_steps,total_samples,sample_size)
```

### Visualization 

#### import visualization module
```
import tf_utils.visualize as vz


num_steps=np.ceil(len(test_df)/batch_size)
total_samples=len(test_df)
vz.plot_confusion_matrix(model,test_ds,num_steps,total_samples,list(species_names.values()))

```
![image](https://user-images.githubusercontent.com/597097/114504489-91a6b000-9c4c-11eb-8433-2fc73fb96e26.png)



