#!/usr/bin/env python
# coding: utf-8

# # Introduction 

# Hi, I’m **Aman**, a Master’s student in Life Science Informatics. This is my deep learning project, where I am implementing a Convolutional Neural Network (CNN) using Keras to identify cells infected by malaria.
# 
# The dataset for training and evaluation is publicly available on Kaggle — it contains microscopic images of both parasitized and uninfected cells, enabling the model to learn distinguishing features through image classification.
# 
# This project aims to:
# 
#          **Explore the fundamentals of deep learning in biomedical image analysis.**
# 
#          **Gain hands-on experience with CNN architectures for real-world healthcare applications.**
# 
#          **Understand data preprocessing, augmentation, and model optimization to improve classification accuracy.**
# 
# Through this project, I am bridging my background in life sciences with my growing expertise in machine learning and artificial intelligence, opening pathways to more advanced research in medical diagnostics and bioinformatics.

# # Modules and Utils 

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Dense , Flatten , Dropout
from tensorflow.keras.metrics import Precision,BinaryAccuracy,Recall
from tensorflow.keras import  layers , models
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imghdr
import os


# # Loading and filtering Data from main dataset

# In[2]:


data_dir = "data"
image_exts =["jpg" , "jpeg" , "png" , "bmp"]
img = cv2.imread(os.path.join(data_dir , "Parasitized" , "C100P61ThinF_IMG_20150918_144104_cell_162.png"))
img.shape


# ### Categories ('Parasitized', 'Uninfected')

# In[3]:


os.listdir(data_dir)


# In[4]:


get_ipython().run_line_magic('pinfo2', 'tf.keras.utils.image_dataset_from_directory')


# **Total Images**

# **Distributing the images in 32 Batches which shuffeling enabled** 

# In Keras, batching images reduces memory usage by processing only a portion of the dataset at once instead of loading it entirely.(because we have more tham 2000 images).
# It speeds up computation because GPUs and TPUs are optimized for handling multiple images in parallel.
# Batching also stabilizes training by averaging gradients over several samples, leading to smoother and more reliable learning.
# Additionally, it allows flexibility in tuning batch size, which can influence model convergence speed and generalization ability.

# In[5]:


data = tf.keras.utils.image_dataset_from_directory(data_dir)
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()


# In[6]:


batch[0].shape  # -> images


# In[7]:


batch[1] # -> Labels


# 
# 
# 
# **We Must Visualize the Image to check which number in our Y array belongs to which label** 

# In[8]:


fig , ax = plt.subplots(ncols = 4 , figsize = (15 , 14))
for index , img in enumerate(batch[0] [:4]):
    ax[index].imshow(img.astype(int))
    ax[index].title.set_text(batch[1][index] )
#Class 0 = Infected / parasitic
#class 1 = uninfected 


# # Data Preprocessing

# ### Normalization of our images
# 

# Normalizing images in a CNN is important because it keeps pixel values within a consistent range, which makes training more stable and faster. It prevents the model from being biased toward brighter or darker images and helps activation functions like ReLU or tanh work efficiently. Normalization also improves generalization, allowing the CNN to learn features that perform well on new, unseen data.
# 
# Here I have used map function to Normalize each Image. Notice that "y" is left as it is, Thats our labels.

# In[9]:


data = data.map(lambda x , y : (x / 255 , y ))
normalized_iter = data.as_numpy_iterator()

print(f"Max Pixel Value {normalized_iter.next()[0].max()} ") #-> higest pixel
print(f"Min Pixel Value {normalized_iter.next()[0].min()} ") # ->lowest pixel



# In[10]:


batch = normalized_iter.next()
fig , ax = plt.subplots(ncols = 4 , figsize = (15 , 14))
for index , img in enumerate(batch[0] [:4]):
    ax[index].imshow(img)
    ax[index].title.set_text(batch[1][index] )
#Class 0 = Infected / parasitic
#class 1 = uninfected 


# In[11]:


batch[1] # -> labels


# # Data Splitting

# In[12]:


print(f"   Total batches {(len(data))} ") # - > total batches
print(f"   batch shape {batch[0].shape}") # - > images in in one batch and channels
total_images = len(data) * batch[0].shape[0]
print(f"   Total Images {total_images}")


# **Splitting data In 70% training , 20% Val and 10% testin**g 

# In[13]:


training_images = int((len(data)* 0.7))
testing_images = int((len(data)* 0.1))
validation_images = int((len(data)* 0.2))
print(training_images + testing_images +  validation_images)
print(training_images ,testing_images ,  validation_images )


# ### train test and val partition

# take(n): Selects the first n elements from a dataset.
# 
# skip(n): Skips the first n elements and returns the rest.
# 
# Used together, they help slice or subset a dataset efficiently in TensorFlow.
# 
# https://www.tensorflow.org/datasets/overview

# In[14]:


train = data.take(training_images)
test = data.skip(training_images).take(validation_images)
val = data.skip(validation_images + training_images).take(testing_images)


# In[15]:


print(f"Training data total {len(train )}")
print(f"Testing data total {len(test)} ")
print(f"Validation data total {len(val)} ")


# # Deep Learning Model

# ### Training

# Here I am using a Sequential CNN model where layers are stacked one after another.
# 
# The model has three convolutional layers with a kernel size of 3x3 to extract features from the input images.
# 
# Each convolutional layer is followed by a MaxPooling2D layer to reduce the spatial dimensions and keep important features.
# 
# After the convolutional layers, the Flatten layer converts the 3D feature maps into a 1D vector.
# 
# The Dense layers process these features to learn complex patterns and make predictions.
# 
# The final layer uses a sigmoid activation to output a probability for binary classification (infected vs. healthy cells).

# In[20]:


cnn = models.Sequential([
    #cnn
    #conv layer 1 
        layers.Conv2D(filters = 32 , kernel_size= (3,3)  , activation= 'relu' , input_shape=(256 ,256 ,3)),
        layers.MaxPooling2D((2,2)),
    #conv layer 2

        layers.Conv2D(filters = 32, kernel_size= (3,3)  , activation= 'relu' ),
        layers.MaxPooling2D((2,2)),
    #conv layer 3

        layers.Conv2D(filters = 16, kernel_size= (3,3)  , activation= 'relu' ),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid' )    
    ])



# In[21]:


cnn.compile(optimizer='adam', 
            loss='binary_crossentropy',
            metrics=['accuracy'])
cnn.summary()


# In[24]:


hist = cnn.fit(train, epochs=3 , validation_data= val)


# In[26]:


print(hist.history.keys())


# Model is performing very well Actually 

# ### Model Evaluation

# In[27]:


import plotly.graph_objects as go

acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
epochs = list(range(1, len(acc)+1))

fig = go.Figure()
fig.add_trace(go.Scatter(x=epochs, y=acc, mode='lines+markers', name='Train Accuracy'))
fig.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines+markers', name='Validation Accuracy'))

fig.update_layout(
    title='Training vs Validation Accuracy',
    xaxis_title='Epoch',
    yaxis_title='Accuracy',
    template='plotly_dark'
)
fig.show()


# In[28]:


loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = list(range(1, len(acc)+1))

fig = go.Figure()
fig.add_trace(go.Scatter(x=epochs, y=loss , mode='lines+markers', name='Train loss'))
fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines+markers', name='Validation loss'))

fig.update_layout(
    title='Training vs Validation loss',
    xaxis_title='Epoch',
    yaxis_title='loss',
    template='plotly_dark'
)
fig.show()


# In[29]:


pre = Precision()
re = Recall()
acc= BinaryAccuracy()

len(test)


# ### Predicting all 172 testing images in the batch

# In[30]:


for batch in test.as_numpy_iterator():
    X,y = batch
    yPred = cnn.predict(X)
    pre.update_state(y , yPred)
    re.update_state(y , yPred)
    acc.update_state(y , yPred)


# In[33]:


print(f"precision {pre.result().numpy()}")
print(f"recall    {re.result().numpy()}")
print(f"accuracy  {acc.result().numpy()}")


# ### Testing

# ### Testing Image 1

# In[84]:


img_test = cv2.imread("infected testing.png")
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

plt.imshow(img_test)
plt.axis("off")

resized = tf.image.resize(img_test, (256, 256))
plt.imshow(resized.numpy().astype("uint8"))
plt.axis("off")
plt.show()
print(resized.shape)

predictor = cnn.predict(np.expand_dims(resized /255 , 0)) # -> Normalizing the testing image (important step)
print(predictor)

if predictor > 0.5 :
    print ("Cell is not infected ")
else :
    print("cell is infected ")


# ### Testing Image 2

# In[85]:


img_test2 = cv2.imread("31-researchersm.jpg")
img_test2 = cv2.cvtColor(img_test2, cv2.COLOR_BGR2RGB)

plt.imshow(img_test2)
plt.axis("off")

resized2 = tf.image.resize(img_test2, (256, 256))
plt.imshow(resize2.numpy().astype("uint8"))
plt.axis("off")
plt.show()
print(resized2.shape)


predictor = cnn.predict(np.expand_dims(resized2 /255 , 0))
print(predictor)

if predictor > 0.5 :
    print ("Cell is not infected ")
else :
    print("cell is infected ")


# # The trained model can now predict images of RBCs and identify whether they are infected or not.

# ## Uses and Benefits of the Model:
# 
# **Malaria Detection**:
#         Automatically identifies RBCs infected by the malaria parasite.
# 
# **Faster Diagnosis**:
#         Reduces the time needed for manual microscopic examination of blood samples.
# 
# **High Accuracy**:
#         Detects subtle signs of infection that might be missed by human observers.
# 
# **Scalable Screening**:
#         Can process large volumes of blood smear images quickly, useful in hospitals and remote clinics.
# 
# **Decision Support for Clinicians**:
#         Assists doctors and lab technicians by providing preliminary screening results.
# 
# **Research Applications**:
#         Useful for studying malaria prevalence, treatment effectiveness, and infection trends.
# 
# **Resource Efficiency**:
#         Reduces dependence on highly trained personnel for routine screening in high-risk areas.

# <div style="font-size: 10px; color: #666; text-align: right;">This notebook was converted with <a href="https://convert.ploomber.io">convert.ploomber.io</a></div>
