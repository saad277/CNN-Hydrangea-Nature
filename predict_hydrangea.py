

import os

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from numpy import set_printoptions

#Initial Status

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3";
os.chdir(r"D:\dataset hydrangea");

set_printoptions(precision=4,suppress=True)

#Loading Model
filepath="hydrangea_cnn_0.92.h5"
my_model=load_model(filepath);

print(my_model.summary());



#Parameters : Weights and Biases for the model

print("Last Layer Bias is : ");

print(my_model.get_weights()[-1]);


print("Last Layer Weight is : ");

print(my_model.get_weights()[-2]);

#Evaluating test

eval_idg=ImageDataGenerator(rescale=1./255);

eval_g=eval_idg.flow_from_directory(directory=r"D:\dataset hydrangea\data2\eval",
                                    target_size=(100,100),
                                    class_mode="binary",
                                    batch_size=20,
                                    shuffle=False)  #takes data one by one

(eval_loss,eval_acc)=my_model.evaluate_generator(generator=eval_g,steps=1)

print(eval_loss);
print(eval_acc);



#Predict individual images

pred_idg=eval_idg
pred_g=eval_g

pred=my_model.predict_generator(generator=pred_g,steps=1)

#will show predictions for 20
print(pred);


#with file name

print(pred_g.filenames,"\n");
print(pred_g.class_indices,"\n");

print(pred[0:10],"\n");  #for hydrangea
print(pred[10:20],"\n")         #for nature


print(pred[0:10] <0.5,"\n");  #for hydrangea
print(pred[10:20]> 0.5,"\n")         #for nature












