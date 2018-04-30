import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
from PIL import Image
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
import h5py
import numpy as np
from keras.layers import Input,merge,Lambda
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D,AveragePooling2D, Conv2DTranspose
from keras.layers.normalization import *
from keras.optimizers import *
from keras import initializers
import matplotlib.pyplot as plt

import cPickle, random, sys, keras
from keras.models import Model
from functools import partial
normal = partial(initializers.normal, scale=.02)
## load and preprocess the dataset (use FERG for example) ##
batch_size  = 256
num_ep      = 7
num_pp      = 6
epochs      = 1000
img_rows, img_cols = 64, 64
clipvalue   = 20
noise_dim   = 10
c_dim       = num_pp
n_dim       = 10
z_dim       = 128
date        = 2018
#

print ('Loading data...')
f = h5py.File('FERG_64_64_color.mat')

print ('Finished loading....')
f      = f['imdb']
label1 = f['id']
label1 = np.asarray(label1)
label1 -= 1
label2 = f['ep']
label2 = np.asarray(label2)
label2 -= 1
label3 = f['set']
label3 = np.asarray(label3)
FrameNum = f['fn']
FrameNum = np.asarray(FrameNum)
x     =  f['images']
x     = np.asarray(x);
x     = np.transpose(x, [3,2,1,0]) # matlab ordering to python ordering
print('x shape:', x.shape)
idx_train = np.asarray(np.where(label3 == 0))
idx_test  = np.asarray(np.where(label3 == 1))
print('idx_test shape',idx_test.shape)


x_train   = x[idx_train[1,:],:,:,:]
x_test    = x[idx_test[1,:],:,:,:]
y_train1  = label1[:,idx_train[1,:]]
y_test1   = label1[:,idx_test[1,:]]
y_train2  = label2[:,idx_train[1,:]]
y_test2   = label2[:,idx_test[1,:]]

y_test1_ori = y_test1
y_test2_ori = y_test2
x_train = (x_train- 127.5)/127.5
x_test  = (x_test- 127.5)/127.5
x_train = x_train.astype('float16')
x_test  = x_test.astype('float16')

y_train1 = keras.utils.to_categorical(y_train1, num_pp)
y_test1  = keras.utils.to_categorical(y_test1, num_pp)
y_train2 = keras.utils.to_categorical(y_train2, num_ep)
y_test2  = keras.utils.to_categorical(y_test2, num_ep)
###############################
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('label 1 train', y_train1.shape)
print('label 1 test', y_test1.shape)
print('label 2 train', y_train2.shape)
print('label 2 test', y_test2.shape)
#
x_ori = (x - 127.5)/127.5
opt  = RMSprop(lr = 0.0003,decay = 1e-6)
dopt = RMSprop(lr = 0.0003,decay = 1e-6)

epsilon_std  = 1.0
def KL_loss(y_true, y_pred):
    z_mean = y_pred[:, 0:z_dim]
    z_log_var = y_pred[:, z_dim:2 * z_dim]
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(kl_loss)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], z_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp((z_log_var) / 2) * epsilon


############ Build the GAN architecture #################
def model_encoder(z_dim, input_shape, units=512, dropout=0.3):
    k = 5
    x = Input(input_shape)
    h = Conv2D(units/8 , (k, k), strides = (2,2), border_mode='same')(x)
    h = BatchNormalization(momentum=0.8)(h)
    h = Dropout(dropout)(h)
    # h = MaxPooling2D(pool_size=(2, 2))(h)
    h = LeakyReLU(0.2)(h)
    h = Conv2D(units/4, (k, k),  strides = (2,2), border_mode='same')(h)
    h = BatchNormalization(momentum=0.8)(h)
    h = Dropout(dropout)(h)
    # h = MaxPooling2D(pool_size=(2, 2))(h)
    h = LeakyReLU(0.2)(h)
    h = Conv2D(units / 2, (k, k), strides = (2,2), border_mode='same')(h)
    h = BatchNormalization(momentum=0.8)(h)
    h = Dropout(dropout)(h)
    # h = MaxPooling2D(pool_size=(2, 2))(h)
    h = LeakyReLU(0.2)(h)
    h = Conv2D(units , (k, k), strides = (2,2), border_mode='same')(h)
    h = BatchNormalization(momentum=0.8)(h)
    h = Dropout(dropout)(h)
    h = LeakyReLU(0.2)(h)
    # h = AveragePooling2D((6,6))(h)
    h = Flatten()(h)
    # h = Dense(latent_dim, name="encoder_mu")(h)
    mean   = Dense(z_dim, name="encoder_mean")(h)
    logvar = Dense(z_dim, name="encoder_sigma", activation = 'sigmoid')(h)
    # meansigma = Model(x, [mean, logsigma],name='encoder')

    z  = Lambda(sampling, output_shape=(z_dim,))([mean, logvar])
    h2 = keras.layers.concatenate([mean,logvar])
    return Model(x,[z, h2], name = 'Encoder')

def model_decoder(z_dim, c_dim):
    k = 5
    x = Input(shape = (z_dim,))
    auxiliary_c = Input(shape=(c_dim,), name='aux_input_c')
    # auxiliary_z = Input(shape=(n_dim,), name='aux_input_z')
    h = keras.layers.concatenate([x, auxiliary_c])
    h = Dense(4 * 4 * 128, activation = 'relu')(h)
    h = Reshape((4, 4, 128))(h)
    # h = LeakyReLU(0.2)(h)
    h = Conv2DTranspose(units, (k,k),  strides = (2,2),  padding = 'same', activation = 'relu')(h) # 32*32*64
    # h = Dropout(dropout)(h)
    h = BatchNormalization(momentum=0.8)(h)
    # h = LeakyReLU(0.2)(h)
    h = Conv2DTranspose(units/2, (k,k),  strides = (2,2),  padding = 'same', activation = 'relu')(h) # 64*64*64
    # h = Dropout(dropout)(h)
    h = BatchNormalization(momentum=0.8)(h)
    # h = LeakyReLU(0.2)(h)
    h = Conv2DTranspose(units/2, (k,k),  strides = (2,2),  padding = 'same', activation = 'relu')(h) # 8*6*64
    # h = Dropout(dropout)(h)
    h = BatchNormalization(momentum=0.8)(h)

    h = Conv2DTranspose(3, (k,k),  strides = (2,2),  padding = 'same', activation = 'tanh')(h) # 8*6*64
    return Model([x,auxiliary_c], h, name="Decoder")

# #### reload the trained weights to implement the anticipated applications####
input_img   = Input((img_rows,img_cols,3))
z_dim       = 128
units       = 256
ee = 200
auxiliary_c = Input(shape=(c_dim,), name='aux_input_c')
auxiliary_z = Input(shape=(n_dim,), name='aux_input_z')
# generator   = model_generator(z_dim = z_dim, input_shape =(img_rows, img_cols, 1) , units=units, dropout=0.3)
encoder = model_encoder(z_dim = z_dim, input_shape =(img_rows, img_cols, 3) , units=units, dropout=0.3)
encoder.load_weights('trained_weight_1.h5')
encoder.compile(loss = 'binary_crossentropy',optimizer = opt)
encoder.summary()

decoder = model_decoder(z_dim = z_dim, c_dim=c_dim)
decoder.load_weights('trained_weight_2.h5')
decoder.compile(loss = 'binary_crossentropy',optimizer = opt)
decoder.summary()



##### expression morphing #####x
for xx in xrange(0,1):
    idx1 = 4300
    idx2 = 7423
    img1 = np.squeeze(x_ori[idx1, :, :, :])
    img2 = np.squeeze(x_ori[idx2, :, :, :])
    z_1, mean_var_imp = encoder.predict(np.expand_dims(img1, axis=0))
    z_2, mean_var_imp = encoder.predict(np.expand_dims(img2, axis=0))

    plt.figure(figsize=(2, 2))
    img1 =np.squeeze(x_ori[idx1,:,:,:])
    img1 = np.uint8(img1*127.5+127.5)
    image = Image.fromarray(img1, 'RGB')
    image.save('ori_1.tif')
    img2 = np.squeeze(x_ori[idx2,:,:,:])
    img2 = np.uint8(img2*127.5+127.5)
    # plt.imshow(img2)
    image = Image.fromarray(img2, 'RGB')
    image.save('ori_2.tif')
    arr = np.linspace(0.0, 1.0, num=1000)
    for ii in xrange(0,1000):
        c = np.ones((1,))*0
        c = keras.utils.to_categorical(c, num_pp)
        z_interp = z_1*(arr[ii])+z_2*(1.0-arr[ii])
        z_interp = np.reshape(z_interp,(1,z_dim))
        img = decoder.predict([z_interp,c])
        img = np.squeeze(img)
        img = np.uint8(img*127.5+127.5)
        image = Image.fromarray(img, 'RGB')
        image.save('interp_'+str(ii)+'.tif')



# ############### Image impanting ##############
loc = 'bottom'
for pp in xrange(0,1):
    for xx in xrange(0,8):
        idx = 123
        input_img    = np.squeeze(x_ori[idx,:,:,:])
        img = np.uint8(input_img*127.5+127.5)
        image = Image.fromarray(img, 'RGB')
        image.save('original.tif')

        impanted_img = np.squeeze(x_ori[idx,:,:,:])
        impanted_img[40:55,18:47,:] = 0 # mouth blocked
        print('impanted_img',impanted_img.shape)

        z_impanted,mean_var_imp = encoder.predict(np.expand_dims(impanted_img,axis =0))
        c = np.ones((1,))*1
        c = keras.utils.to_categorical(c, num_pp)
        print('c',c)
        img_rec = decoder.predict([z_impanted,c])
        img_rec = np.squeeze(img_rec)

        img = np.uint8(impanted_img*127.5+127.5)
        image = Image.fromarray(img, 'RGB')
        image.save('test_blocked_pp1'+'.tif')

        img = np.uint8(img_rec*127.5+127.5)
        image = Image.fromarray(img, 'RGB')
        image.save('test_rec_pp1'+'.tif')

        impanted_img[40:55,18:47,:] = img_rec[40:55,18:47,:]
        img = np.uint8(impanted_img*127.5+127.5)
        image = Image.fromarray(img, 'RGB')
        image.save('test_replaced_pp1'+'.tif')



#### Generate images without input image ###
def sampling_np( z_mean, z_log_var ):
    epsilon = np.random.normal(loc=0., scale=epsilon_std, size=(z_mean.shape[0], z_dim), )
    return z_mean + np.exp(z_log_var / 2) * epsilon

# mean and variance of the prior distribution #
mean_train_sup  = np.zeros((1,128))
var_train_sup   = np.ones((1,128))

for i in xrange(0,num_pp):
    for xx in xrange(0,100):
        z = sampling_np(mean_train_sup, var_train_sup)
        print(z.shape)
        c = np.ones(1,)*i
        c = keras.utils.to_categorical(c, num_pp)
        img = decoder.predict([z, c])
        img = np.squeeze(img)
        img = np.uint8(img*127.5+127.5)
        image = Image.fromarray(img, 'RGB')
        image.save('synthesis_no_input_'+'pp_'+str(i)+'.tif')