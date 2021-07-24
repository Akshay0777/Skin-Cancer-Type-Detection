import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image

np.random.seed(42)

import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder

skin_df = pd.read_csv('data/HAM10000_metadata.csv')

SIZE=32

## label encoding to numeric values from text
le = LabelEncoder()
le.fit(skin_df['dx'])
LabelEncoder()
print(list(le.classes_))
skin_df['label'] = le.transform(skin_df["dx"]) 
print(skin_df.sample(10))

#   Melanocytic nevi (nv) 0
#   Melanoma (mel) 1
#   Benign keratosis-like lesions (bkl) 2
#   Basal cell carcinoma (bcc) 
#   Actinic keratoses (akiec)
#   Vascular lesions (vas)
#   Dermatofibroma (df)


## Distribution of data into various classes
from sklearn.utils import resample
# print(skin_df['label'].value_counts())

df_0 = skin_df[skin_df['label'] == 0]
df_1 = skin_df[skin_df['label'] == 1]
df_2 = skin_df[skin_df['label'] == 2]
df_3 = skin_df[skin_df['label'] == 3]
df_4 = skin_df[skin_df['label'] == 4]
df_5 = skin_df[skin_df['label'] == 5]
df_6 = skin_df[skin_df['label'] == 6]

n_samples=500 
df_0_balanced = resample(df_0, replace=True, n_samples=n_samples, random_state=42) 
df_1_balanced = resample(df_1, replace=True, n_samples=n_samples, random_state=42) 
df_2_balanced = resample(df_2, replace=True, n_samples=n_samples, random_state=42)
df_3_balanced = resample(df_3, replace=True, n_samples=n_samples, random_state=42)
df_4_balanced = resample(df_4, replace=True, n_samples=n_samples, random_state=42)
df_5_balanced = resample(df_5, replace=True, n_samples=n_samples, random_state=42)
df_6_balanced = resample(df_6, replace=True, n_samples=n_samples, random_state=42)

##  Combined back to a single dataframe
skin_df_balanced = pd.concat([df_0_balanced, df_1_balanced, 
                              df_2_balanced, df_3_balanced, 
                              df_4_balanced, df_5_balanced, df_6_balanced])

##  Check the distribution. All classes should be balanced now.
# print(skin_df_balanced['label'].value_counts())

##  Now time to read images based on image ID from the CSV file
##  This is the safest way to read images as it ensures the right image is read for the right ID
image_path = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join('data/', '*', '*.jpg'))}

print(image_path)
##  Define the path and add as a new column
skin_df_balanced['path'] = skin_df['image_id'].map(image_path.get)
##  Use the path to read images.
skin_df_balanced['image'] = skin_df_balanced['path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE,SIZE))))


print(skin_df_balanced)
#############

##  Convert dataframe column of images into numpy array
X = np.asarray(skin_df_balanced['image'].tolist())
X = X/255.  # Scale values to 0-1
Y=skin_df_balanced['label']  #Assign label values to Y
Y_cat = to_categorical(Y, num_classes=7) #Convert to categorical as this is a multiclass classification problem

# print(Y_cat)

##  Split to training and testing
x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.25, random_state=42)

##  Define the model.

from keras import models,layers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(SIZE, SIZE, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))  #Dropout for regularization
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])

###############

# Train

batch_size = 16 
epochs = 50

history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size = batch_size,
    validation_data=(x_test, y_test),
    verbose=2)

pred = model.predict(x_test)
print(np.argmax(pred[:10],axis=1))
print(y_test[:10])
# result
# [3 1 4 3 0]
# [[0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0.]]



model.save('model_keras.h5')

score = model.evaluate(x_test, y_test)
print('Test accuracy:', score[1])


# Epoch 50/50
# 165/165 - 5s - loss: 0.2185 - acc: 0.9189 - val_loss: 0.8449 - val_acc: 0.7920
