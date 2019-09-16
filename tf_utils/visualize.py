#function to plot n images from dataset
def plot_cifar10_files(dataset,n=5):
  import matplotlib.pyplot as plt
  records=dataset.take(1)
  #print(records)

  for record in records:
    image_batch,label_batch=record
    
    image_batch=image_batch.numpy()
    for i in range(n):
      
      plt.imshow(image_batch[i].astype('uint8'))
      plt.show()
      
#function to get misclssified images
def get_misclassified_images(model,test_ds):
  num_steps=np.ceil(10000/batch_size)
  pred=model.predict(test_ds,steps =num_steps, verbose=1)
  pred2=np.argmax(pred,axis=1)
  wrong_set=[]
  correct_set=[]
  wrong_labels=[]
  true_labels=[]
  wrong_indices=[]
  
  c=0
  y=None
  
  for record in test_ds.take(num_steps):
    if c==0:
      x=record[0].numpy()
      y=record[1].numpy()
      c+=1
    else:
      y= np.vstack((y,record[1].numpy()))
      x= np.vstack((x,record[0].numpy()))
  y=y[:10000]
  pred2=pred2[:10000]
  x=x[:10000]
  for i in range(10000):
    y1=np.argmax(y[i])
    if pred2[i]==y1:
      correct_set.append(x[i])
    else:
      wrong_indices.append(i)
      wrong_labels.append(class_names[pred2[i]])
      true_labels.append(class_names[y1])
      wrong_set.append(x[i])
  
  return wrong_indices, wrong_labels, true_labels, wrong_set  


#function to display images 
import matplotlib.pyplot as plt
def displayRow(images,titles):
  n=len(images)
  m=4-n
  
  if n<4:
    for j in range(m):
      
      dummy_image=(np.ones([32,32,3]))
      dummy_image=dummy_image*255
      images.append(dummy_image)
      titles.append('')
      
      
      
  
  
  fig = plt.figure(1, (13,13))
  
  grid = ImageGrid(fig, 111,  
                 nrows_ncols=(1,len(images)),  
                 axes_pad=1,label_mode="1"  
                 )
  
  for i in range(len(images)):
    grid[i].imshow(images[i].astype('uint8'))
    grid[i].set_title(titles[i])
    grid[i].axis('off')
  plt.show()
  
  
  display(HTML("<hr size='5' color='black' width='100%' align='center' />"))

#function to grop misclassified records in rows
from mpl_toolkits.axes_grid1 import ImageGrid

from IPython.core.display import display, HTML

def plot_misclassified_images(wrong_indices,wrong_labels,true_labels,wrong_set,num_images=50):
  
  heder="<h2 align='center'>First "+str(num_images)+" misclassified images</h2><hr size='5' color='black' width='100%' align='center' />"
  display(HTML(heder))
  
  for i in range(0,num_images,4):
    images=[]
    titles=[]
    
    for j in range(4):
      
      if (i+j)<num_images:
        
        images.append(wrong_set[i+j])
        title=str(wrong_indices[i+j])+':'+true_labels[i+j]+'\n predicted as \n'+wrong_labels[i+j]
    
        titles.append(title)
    
    
    
    
    displayRow(images,titles)
