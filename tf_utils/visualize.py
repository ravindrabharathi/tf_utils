from tf_utils.data import *
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

      
#function to get misclssified images
@tf.function
def get_misclassified_images(model,test_ds):
  num_steps=np.ceil(len(list(test_ds))/batch_size)
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
  y=y[:num_steps]
  pred2=pred2[:num_steps]
  x=x[:10000]
  for i in range(num_steps):
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
      
      dummy_image=(np.ones([256,256,3]))
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


# function to plot confusion matrix
def plot_confusion_matrix(model,test_ds):  
  num_steps=np.ceil(len(list(test_ds))/batch_size) ## not a good way to get length , tf 2.3 has dataset.cardinality().numpy()
  pred=model.predict(test_ds,steps =num_steps, verbose=1)
  pred2=np.argmax(pred,axis=1)
  
  c=0
  for record in test_ds.take(num_steps):
    if c==0:
      x=record[0].numpy()
      y=record[1].numpy()
      c+=1
    else:
      y= np.vstack((y,record[1].numpy()))
      x= np.vstack((x,record[0].numpy()))

  y=y[:num_steps]
  pred2=pred2[:num_steps]
  x=x[:num_steps]
  y1=np.argmax(y, axis=1)   
  
  c_matrix = metrics.confusion_matrix(y1, pred2)  
  
  
  fig, ax= plt.subplots()
  fig.set_figheight(10)
  fig.set_figwidth(10)
  title='confusion matrix'
  im = ax.imshow(c_matrix, interpolation='nearest', cmap=plt.cm.Oranges)
  
  ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
 
  ax.set(xticks=np.arange(c_matrix.shape[1]),
           yticks=np.arange(c_matrix.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

  fmt = 'd' 
  thresh = c_matrix.max() / 2.
  for i in range(c_matrix.shape[0]):
      for j in range(c_matrix.shape[1]):
          ax.text(j, i, format(c_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if c_matrix[i, j] > thresh else "black")
  fig.tight_layout()    

def visualize_data(data_df):
  print('Class Mapping\n-----------------')
  class_map=data_df[['Label','Species']]
  class_map=class_map.drop_duplicates()
  class_map=class_map.sort_values(by=['Label']).reset_index(drop=True)
  print(class_map)
  print('---------------------\n')

  print('Count per Species\n-----------------')

  vc=data.Species.value_counts().rename_axis('Species').reset_index(name='Total')
  print(vc)
  print('---------------------\n')
  labels=vc.Species
  fig, ax = plt.subplots(1, 1)
  species_hist = ax.hist(data_df.Label,bins=9,alpha=0.6,histtype='bar',ec='black',orientation='horizontal',)
  
  
  plt.show()
  return class_map


def show_image_sample(data_df,images_dir='./images'):
  classes=['Chinee apple', 'Lantana', 'Parkinsonia', 'Parthenium', 'Prickly acacia', 'Rubber vine', 'Siam weed','Snake weed', 'Negative']
  fig, ax = plt.subplots(9,5,sharex=True,figsize=(8, 14))
  for i in range(9):
      
      for j in range(5):
            
            
            if ( j==0 and (i<9) ):
                plt.setp(ax[i,j].get_xticklabels(), visible=False)
                ax[i,j].tick_params(axis='x',which='both',bottom=False, top=False,labelbottom=False)
                ax[i,j].tick_params(axis='y',which='both',left=False, right=False,labelleft=False)
                
            else:
                ax[i,j].axis('off')  
  for i in range(len(classes)):
    clazz=classes[i]
    ax[i,0].set_ylabel(clazz,fontsize=12)
    df1=data_df[data_df.Species==clazz]
    files=df1.head().Filename.to_list()
    for j in range(len(files)):
      filename=files[j]
      if images_dir !='':
        filename=os.path.join(images_dir,filename)
      img = Image.open(filename)
      img = np.array(img.resize((256,256)))
      ax[i,j].imshow(img)
      
  plt.tight_layout(pad=2.0) 
  plt.show()

  
