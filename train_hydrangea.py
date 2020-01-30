

import tensorflow as tf

import os
from os import environ , chdir
from keras.preprocessing.image import ImageDataGenerator
from keras import Input,models,layers,optimizers,callbacks          #input was necessary to start a tensor


environ["TFF_CPP_MIN_LOG_LEVEL"]="3";

chdir(r'D:\dataset hydrangea');


#Setting Image and Data Generator
#idg=image data generator
      
train_idg=ImageDataGenerator(rescale=1./255,zoom_range=[1.0,1.25],width_shift_range=0.1,height_shift_range=0.1,fill_mode="reflect");



#image data generator settings applied to your train images
train_g=train_idg.flow_from_directory(directory=r"data2\train",
                                      target_size=(100,100),        #target size will rescale images to 100x100
                                      class_mode="binary",          #this will tell yes or no for classification
                                      batch_size=125,
                                      shuffle=True);




valid_ig=ImageDataGenerator(rescale=1./255);

valid_g=train_idg.flow_from_directory(directory=r"data2\valid",
                                      target_size=(100,100),        
                                      class_mode="binary",         
                                      batch_size=125,
                                      shuffle=True);



#CNN Architecture

my_model=models.Sequential();

#1st layer

my_model.add(layers.Conv2D(filters=16,kernel_size=(2,2),strides=(1,1),input_shape=(100,100,3)));

my_model.add(layers.Activation("relu"));

#2nd Layer

my_model.add(layers.Conv2D(filters=16,kernel_size=(2,2),strides=(1,1)));     #no need to specify shape again

my_model.add(layers.Activation("relu"));

#......

my_model.add(layers.MaxPooling2D(pool_size=(2,2)));

my_model.add(layers.Dropout(rate=0.4));





#3rd layer

my_model.add(layers.Conv2D(filters=16,kernel_size=(2,2),strides=(1,1)));

my_model.add(layers.Activation("relu"));

#4th Layer

my_model.add(layers.Conv2D(filters=16,kernel_size=(2,2),strides=(1,1)));     #no need to specify shape again

my_model.add(layers.Activation("relu"));

#......

my_model.add(layers.MaxPooling2D(pool_size=(2,2)));

my_model.add(layers.Dropout(rate=0.4));




#5th layer

my_model.add(layers.Conv2D(filters=16,kernel_size=(2,2),strides=(1,1)));

my_model.add(layers.Activation("relu"));

my_model.add(layers.MaxPooling2D(pool_size=(2,2)));

#6th Layer

my_model.add(layers.Conv2D(filters=16,kernel_size=(2,2),strides=(1,1)));     #no need to specify shape again

my_model.add(layers.Activation("relu"));

#......
my_model.add(layers.MaxPooling2D(pool_size=(2,2)));

my_model.add(layers.Flatten());

my_model.add(layers.Dropout(rate=0.4));

#......

my_model.add(layers.Dense(units=10));

my_model.add(layers.Activation("relu"));


my_model.add(layers.Dense(units=1));

my_model.add(layers.Activation("sigmoid"));


print(my_model.summary());

#model lost function and optimizer method


compile=my_model.compile(optimizer=optimizers.sgd(lr=0.15),loss="binary_crossentropy",metrics=["accuracy"]);      #lr = learning rate



#Setting Callbacks

check_p=callbacks.ModelCheckpoint(filepath="hydrangea_cnn_{val_accuracy:.2f}.h5",monitor="val_accuracy",verbose=1,save_best_only=True,save_weights_only=False);

reduce_lr=callbacks.ReduceLROnPlateau(monitor="val_accuracy",factor=0.95,patience=5,verbose=1,cooldown=2);                             #this will increase learning rate accuracy 

callback_list=[check_p,reduce_lr];



#Training Options

fit=my_model.fit_generator(generator=train_g,steps_per_epoch=22,epochs=50,verbose=1,callbacks=callback_list,validation_data=valid_g,validation_steps=4);



#saving model

my_model.save(filepath=r"hydrangea_cnn.h5",overwrite=True)


print("Xxxx");



















      


