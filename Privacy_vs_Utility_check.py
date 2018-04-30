from __future__ import print_function
from keras.layers import Dense, Dropout, Flatten,GlobalAveragePooling2D, Activation, Input, Reshape
from keras.optimizers import SGD,Adagrad,RMSprop, Adam
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping,History, LearningRateScheduler
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.layers.advanced_activations import LeakyReLU
import time
######################################### dataset parameters ###########################################
data_augmentation = False
# #
batch_size  = 256
num_ep      = 7
num_pp      = 6
epochs      = 10
img_rows, img_cols = 64, 64
clipvalue   = 20
noise_dim   = 10
c_dim       = num_pp
n_dim       = 10
date        = 1205
####### Choose your task #######
# Task        = 'ExpRec'
Task        = 'Identification'
###############################

### Load your data, either f(I) or I ###

########################################


############ CNN model(for I) #################
d0 = Input((x_train.shape[1:]))
# x0 = Dense(img_rows*img_cols*1, activation = 'relu')(d0)
# x0 = Reshape((img_rows,img_cols,1))(x0)
x = Conv2D(32, (5,5), padding = 'same', name = 'id_conv1')(d0)
x = LeakyReLU(0.2)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), padding='same', strides = (2,2))(x)
x = Dropout(0)(x)

x = Conv2D(32, (5,5), padding = 'same', name = 'id_conv2')(x)
x = LeakyReLU(0.2)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), padding='same', strides = (2,2))(x)
x = Dropout(0)(x)

x = Conv2D(32, (3,3), padding = 'same', name = 'id_conv3')(x)
x = LeakyReLU(0.2)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), padding='same', strides = (2,2))(x)
x = Dropout(0)(x)

x = Conv2D(32, (3,3),  padding = 'same', name = 'id_conv4')(x)
x = LeakyReLU(0.2)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), padding='same', strides = (2,2))(x)
x = Dropout(0)(x)

x = Flatten()(x)
x = Dense(32,  name = 'id_dense1')(x)
x = LeakyReLU(0.2)(x)
x = Dropout(0.5)(x)
if Task == 'Identification':
    output_main = Dense(num_pp, activation = 'softmax', name = 'id_dense2')(x)
elif Task == 'ExpRec':
    output_main = Dense(num_ep, activation = 'softmax', name = 'id_dense2')(x)

model = Model(d0,output_main)


#### ANN model(for f(I))####
d0 = Input((x_train.shape[1:]))
# d0 = Flatten()(d0)
x = Dense(128,  name = 'id_dense1')(d0)
x = LeakyReLU(0.2)(x)
x = Dropout(0.25)(x)
x = Dense(128,  name = 'id_dense2')(x)
x = LeakyReLU(0.2)(x)
x = Dropout(0.25)(x)
x = Dense(128,  name = 'id_dense3')(x)
x = LeakyReLU(0.2)(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
if Task == 'Identification':
    output_main = Dense(num_pp, activation = 'softmax', name = 'id_dense4')(x)
elif Task == 'ExpRec':
    output_main=Dense(num_ep, activation = 'softmax', name = 'id_dense4')(x)
model = Model(d0,output_main)


#############################################
# initiate optimizer
# opt = keras.optimizers.SGD(lr=0.001, decay=1e-6)
adam = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-8)
sgd  = SGD(lr=0.005, momentum=0.9, decay=1e-7, nesterov=True)
model.compile(loss= ['categorical_crossentropy'], #'categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
model.summary()
# plot_model(model, to_file = 'ANN_model.png')
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')



# model_filename = '/home/vivo/Dropbox/Encrypted_video_classification/ImageSharing-master/DES_encrypted_ixmas/model/0721.{epoch:02d}-{val_loss:.2f}.h5'
callbacks = [
    EarlyStopping(monitor='loss',
                  patience=100,
                  verbose=1,
                  mode='auto'),
    ModelCheckpoint(filepath='/loc.h5',
                    monitor='val_acc',
                    save_best_only=True),
    History(),
]
# plot_model(model, to_file='de_id_mse_model.png',show_shapes = True)
start = time.time()
count0 = 0

history = model.fit(x_train, y_train1,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test ,y_test1),
                    shuffle=True, callbacks = callbacks)
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title(Task)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()
plt.savefig('/media/vivo/New Volume/FERG_DB_256/VAE_GAN_stats/vaegan_FERG_color_train_on_ori_'+Task+'_'+str(date)+'.png')
plt.close()


finish_time = time.time()
print("Elapsed: %s " % (finish_time - start) )

