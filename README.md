# tf_utils  (Active Learning)
Utility package for using active learning strategies for labelling image datesets and training CNN using tensorflow2, TFRecords, tf.data
Supports the following
1. DroughtWatch Dataset from Weights and Biases 
2. Retreve Data as tf.data.TFRecordDataset
3. Image Augmentation (random-pad_crop, flip-left_right, cutout) of Image Batches
4. Active Learning data points selection for labelling using Least confidence sampling , Margin sampling (Entropy sampling to be added) 
5. Random sampling method for getting a baseline is supported 
6. Plot RGB images from Dataset 
7. Plot Confusion Matrix

### Installation
```
!pip install --upgrade git+https://github.com/ravindrabharathi/tf_utils@active_learning_drought_watch
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




