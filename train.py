import os, random

os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
import h5py
import numpy as np
from keras.utils import plot_model

from keras.layers import Input, merge, Lambda
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, AveragePooling2D, \
    Conv2DTranspose
from keras.layers.normalization import *
from keras.optimizers import *
from keras import initializers
import matplotlib.pyplot as plt
import cPickle, random, sys, keras
from keras.models import Model
from tqdm import tqdm
import time
import os, sys
from functools import partial

normal = partial(initializers.normal, scale=.02)

## load and preprocess the dataset ##
batch_size = 256
num_ep = 7
num_pp = 8
epochs = 300
img_rows, img_cols = 64, 64
c_dim = num_pp
date = 2018

print ('Loading data...')
f = h5py.File('/data')
print ('Finished loading....')


epsilon_std = 1
def sampling_np(args):
    z_mean, z_log_var = args
    epsilon = np.random.normal(loc=0., scale=epsilon_std, size=(z_mean.shape[0], z_dim), )
    return z_mean + np.exp(z_log_var / 2) * epsilon


def generate_dataset(ee):
    ## save to numpyz###############
    c = np.random.randint(num_pp, size=x_train.shape[0])
    c_train = keras.utils.to_categorical(c, num_pp)
    c = np.random.randint(num_pp, size=x_test.shape[0])
    c_test = keras.utils.to_categorical(c, num_pp)

    [z_train, mean_var_train] = encoder.predict(x_train)
    encoded_xtrain = decoder.predict([z_train, c_train])

    [z_test, mean_var_test] = encoder.predict(x_test)
    encoded_xtest = decoder.predict([z_test, c_test])

    np.savez('/Z_' + str(date) + 'epoch'+str(ee)+'_64_64_VAE_GAN_labelfull_v2.npz',
             encoded_xtrain, y_train1, y_train2, c_train, encoded_xtest, y_test1, y_test2, c_test)
    np.savez('/X_' + str(date) + 'epoch'+str(ee)+ '_fi_512_VAE_GAN_labelfull_v2.npz',
             z_train, y_train1, y_train2, c_train, z_test, y_test1, y_test2, c_test)

opt  = RMSprop(lr=0.0003, decay=1e-6)
dopt = RMSprop(lr=0.0003, decay=1e-6)


def KL_loss(y_true, y_pred):
    z_mean = y_pred[:, 0:z_dim]
    z_log_var = y_pred[:, z_dim:2 * z_dim]
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(kl_loss)


def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], z_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(K.square(z_log_sigma) / 2) * epsilon


def model_encoder(z_dim, input_shape, units=512, dropout=0.3):
    k = 8
    x = Input(input_shape)
    h = Conv2D(units / 8, (k, k), strides=(2, 2), border_mode='same')(x)
    h = BatchNormalization(momentum=0.8)(h)
    h = Dropout(dropout)(h)
    # h = MaxPooling2D(pool_size=(2, 2))(h)
    h = LeakyReLU(0.2)(h)
    h = Conv2D(units / 4, (k, k), strides=(2, 2), border_mode='same')(h)
    h = BatchNormalization(momentum=0.8)(h)
    h = Dropout(dropout)(h)
    # h = MaxPooling2D(pool_size=(2, 2))(h)
    h = LeakyReLU(0.2)(h)
    h = Conv2D(units / 2, (k, k), strides=(2, 2), border_mode='same')(h)
    h = BatchNormalization(momentum=0.8)(h)
    h = Dropout(dropout)(h)
    # h = MaxPooling2D(pool_size=(2, 2))(h)
    h = LeakyReLU(0.2)(h)
    h = Conv2D(units, (k, k), strides=(2, 2), border_mode='same')(h)
    h = BatchNormalization(momentum=0.8)(h)
    h = Dropout(dropout)(h)
    h = LeakyReLU(0.2)(h)
    h = Flatten()(h)
    mean = Dense(z_dim, name="encoder_mean")(h)
    logvar = Dense(z_dim, name="encoder_sigma", activation='sigmoid')(h)

    z = Lambda(sampling, output_shape=(z_dim,))([mean, logvar])
    h2 = keras.layers.concatenate([mean, logvar])
    return Model(x, [z, h2], name='Encoder')


def model_decoder(z_dim, c_dim):
    k = 8
    x = Input(shape=(z_dim,))
    auxiliary_c = Input(shape=(c_dim,), name='aux_input_c')
    h = keras.layers.concatenate([x, auxiliary_c])
    h = Dense(4 * 4 * 128, activation='relu')(h)
    h = Reshape((4, 4, 128))(h)
    # h = LeakyReLU(0.2)(h)
    h = Conv2DTranspose(units, (k, k), strides=(2, 2), padding='same', activation='relu')(h)  # 32*32*64
    # h = Dropout(dropout)(h)
    h = BatchNormalization(momentum=0.8)(h)
    # h = LeakyReLU(0.2)(h)
    # h = UpSampling2D(size=(2, 2))(h)
    h = Conv2DTranspose(units / 2, (k, k), strides=(2, 2), padding='same', activation='relu')(h)  # 64*64*64
    # h = Dropout(dropout)(h)
    h = BatchNormalization(momentum=0.8)(h)
    # h = LeakyReLU(0.2)(h)
    # h = UpSampling2D(size=(2, 2))(h)
    h = Conv2DTranspose(units / 2, (k, k), strides=(2, 2), padding='same', activation='relu')(h)  # 8*6*64
    # h = Dropout(dropout)(h)
    h = BatchNormalization(momentum=0.8)(h)

    h = Conv2DTranspose(3, (k, k), strides=(2, 2), padding='same', activation='tanh')(h)  # 8*6*64
    return Model([x, auxiliary_c], h, name="Decoder")


################################################ Build the discrminator ###########################################################################

input_shape = (img_rows, img_cols, 3)
loss_weights_1= Input(shape=(1,), name='disc_1')
loss_weights_2= Input(shape=(1,),name='disc_2')
loss_weights_3= Input(shape=(1,),name='disc_3')
targets1  = Input(shape = (1,),name='disc_4')
targets2  = Input(shape = (num_pp,),name='disc_5')
targets3  = Input(shape = (num_ep,),name='disc_6')
d_input   = Input(input_shape,name='disc_7')
rep_field = 8
x = Conv2D(32, (rep_field, rep_field), strides=(2, 2), padding='same', name='id_conv1')(d_input)
# x = BatchNormalization(momentum=0.9)(x)
x = LeakyReLU(0.2)(x)
# x  = AveragePooling2D((2, 2), padding='same')(x)
# x  = Dropout(0.3)(x)

x = Conv2D(64, (rep_field, rep_field), strides=(2, 2), padding='same', name='id_conv2')(x)
# x = BatchNormalization(momentum=0.9)(x)
x = LeakyReLU(0.2)(x)
# x  = AveragePooling2D((2, 2), padding='same')(x)
# x  = Dropout(0.3)(x)

x = Conv2D(128, (rep_field, rep_field), strides=(2, 2), padding='same', name='id_conv3')(x)
# x = BatchNormalization(momentum=0.9)(x)
x = LeakyReLU(0.2)(x)
# x  = AveragePooling2D((2, 2), padding='same')(x)
# x  = Dropout(0.3)(x)

x = Conv2D(256, (rep_field, rep_field), strides=(2, 2), padding='same', name='id_conv4')(x)
# x = BatchNormalization(momentum=0.9)(x)
x = LeakyReLU(0.2)(x)

x = Flatten()(x)
x = Dense(256, name='ds')(x)
x = LeakyReLU(0.2)(x)
x = Dropout(0.5)(x)
output_binary     = Dense(1, activation='sigmoid', name='bin_real')(x)
output_identity   = Dense(num_pp, activation='softmax', name='id_real')(x)
output_expression = Dense(num_ep, activation='softmax', name='exp_real')(x)

discriminator = Model([d_input, loss_weights_1, loss_weights_2,loss_weights_3, targets1, targets2, targets3], [output_binary, output_identity, output_expression])

from keras import losses

loss =loss_weights_1*losses.binary_crossentropy(targets1,output_binary) + \
      loss_weights_2*losses.categorical_crossentropy(targets2,output_identity)+ \
      loss_weights_3*losses.categorical_crossentropy(targets3,output_expression)
discriminator.add_loss(loss)
discriminator.compile( optimizer=dopt, loss = None)
discriminator.summary()
print (discriminator.metrics_names)
plot_model(discriminator, to_file = '/media/vivo/New Volume/FERG_DB_256/stats/disc_0605_model.png')

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


make_trainable(discriminator, False)
discriminator.trainable = False

# #### Build GAN model ####
z_dim = 128
units = 256
GANloss_weights_vae = Input(shape = (1,))
GANtargets_vae  = Input(shape = (z_dim*2,))

ee = 100

auxiliary_c = Input(shape=(c_dim,), name='aux_input_c')
encoder = model_encoder(z_dim=z_dim, input_shape=(img_rows, img_cols, 3), units=units, dropout=0.3)
# encoder.load_weights('/media/vivo/New Volume/FERG_DB_256/model/VAEGAN_5th_encoder_MUG_8pp_real_' + str(date)+'epochs'+str(ee)+ '.h5')
# encoder.compile(loss='binary_crossentropy', optimizer=opt)
# encoder.summary()

decoder = model_decoder(z_dim=z_dim, c_dim=c_dim)
# decoder.load_weights('/media/vivo/New Volume/FERG_DB_256/model/VAEGAN_5th_decoder_MUG_8pp_real_' + str(date)+'epochs'+str(ee)+ '.h5')
# decoder.compile(loss='binary_crossentropy', optimizer=opt)
# decoder.summary()

### Generate Image set ###
# generate_dataset(ee=ee)
###


### GAN formulation ###
[z, mean_var] = encoder(d_input)
xpred = decoder([z, auxiliary_c])
output_binary, output_identity, output_expression = discriminator([xpred, loss_weights_1, loss_weights_2,loss_weights_3, targets1, targets2, targets3])
GAN = Model([d_input, auxiliary_c, GANloss_weights_vae, loss_weights_1,loss_weights_2,loss_weights_3, GANtargets_vae, targets1, targets2, targets3],\
            [mean_var, output_binary, output_identity, output_expression])

GANloss = GANloss_weights_vae*KL_loss(GANtargets_vae, mean_var) + \
          loss_weights_1*losses.binary_crossentropy(targets1,output_binary) + \
          loss_weights_2*losses.categorical_crossentropy(targets2, output_identity)+ \
          loss_weights_3*losses.categorical_crossentropy(targets3, output_expression)
GAN.add_loss(GANloss)
GAN.compile(optimizer = opt, loss = None)
GAN.summary()
print (GAN.metrics_names)


# plot_model(GAN, to_file = 'GAN_model.png')

def plotGeneratedImages(epoch, idx=0, examples=10, dim=(10, 10), figsize=(10, 10)):
    n = num_pp  # how many digits we will display
    pp_avg = 4500
    plt.figure(figsize=(16, 4))


    sample = x_ori[idx:idx + n, :, :, :]
    c = np.asarray([0, 1, 2, 3, 4, 5, 6, 7])
    c = keras.utils.to_categorical(c, num_pp)

    [z, mean_var] = encoder.predict(sample)
    generated_images = decoder.predict([z, c])

    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        ori = sample[i].reshape(img_rows, img_cols, 3)
        ori = np.uint8(ori * 127.5 + 127.5)
        plt.imshow(ori)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        rec = generated_images[i].reshape(img_rows, img_cols, 3)
        rec = np.uint8(rec * 127.5 + 127.5)
        plt.imshow(rec)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # Path to be created
    plt.savefig(path + '/GAN_MUG_results_' + str(date) + '_generated_image_epoch_%d.tif' % epoch)
    plt.close()


def train_for_n(nb_epoch=50000, plt_frq=25, BATCH_SIZE=256):
    batchCount = x_train.shape[0] / BATCH_SIZE
    for ee in xrange(1, nb_epoch + 1):
        print '-' * 15, 'Epoch %d' % ee, '-' * 15
        plotGeneratedImages(epoch=ee + 40, idx=75)
        # val_bin_acc, val_id_acc, val_ep_acc = val_test()
        for e in tqdm(range(batchCount)):
            # for didx in xrange(0,k):
            idx = random.sample(range(0, x_train.shape[0]),
                                BATCH_SIZE)  # train discriminator twice more than the generator
            image_batch = x_train[idx, :, :, :]  # real data
            c = np.random.randint(num_pp, size=BATCH_SIZE)
            c = keras.utils.to_categorical(c, num_pp)

            [z, mean_var] = encoder.predict(image_batch)
            generated_images = decoder.predict([z, c])

            y1_batch = y_train1[idx, :]
            y2_batch = y_train2[idx, :]

            # generated_images = generator.predict([image_batch, c_, z])
            y0_dist_real = np.random.uniform(0.9, 1.0, size=[BATCH_SIZE, 1])
            y0_dist_fake = np.random.uniform(0, 0.1, size=[BATCH_SIZE, 1])



            make_trainable(discriminator, True)
            discriminator.trainable = True
            loss_weights_1 = np.ones(shape = (batch_size,))*1/4.0
            loss_weights_2 = np.ones(shape = (batch_size,))*1/2.0
            loss_weights_3 = np.ones(shape = (batch_size,))*1/4.0
            d_loss_real = discriminator.train_on_batch([image_batch, loss_weights_1, loss_weights_2, loss_weights_3,y0_dist_real, y1_batch, y2_batch],y= None)
            loss_weights_1 = np.ones(shape=(batch_size,))*1.0
            loss_weights_2 = np.ones(shape=(batch_size,)) * 0
            loss_weights_3 = np.ones(shape=(batch_size,)) * 0
            d_loss_fake = discriminator.train_on_batch([generated_images,loss_weights_1,loss_weights_2,loss_weights_3, y0_dist_fake, c, y2_batch], y = None)


            make_trainable(discriminator, False)
            discriminator.trainable = False
            for ii in xrange(0, 2):
                idx = random.sample(range(0, x_train.shape[0]),
                                    BATCH_SIZE)  # train discriminator twice more than the generator
                image_batch = x_train[idx, :, :, :]  # real data
                c = np.random.randint(num_pp, size=BATCH_SIZE)
                c = keras.utils.to_categorical(c, num_pp)

                mean_var_ref = np.ones((BATCH_SIZE, z_dim * 2))
                y1_batch = y_train1[idx, :]
                y2_batch = y_train2[idx, :]

                y0_batch = np.ones((BATCH_SIZE, 1)) #0.002, 0.09, 0.8, 0.108
                GANloss_weights_vae = np.ones(shape = (batch_size,))*0.002
                loss_weights_1 = np.ones(shape = (batch_size,))*0.078
                loss_weights_2 = np.ones(shape = (batch_size,))*0.8
                loss_weights_3 = np.ones(shape = (batch_size,))*0.12
                g_loss = GAN.train_on_batch([image_batch, c, GANloss_weights_vae, loss_weights_1, loss_weights_2, loss_weights_3, mean_var_ref, y0_batch, c, y2_batch], y = None)


        if ee % 25 == 0:
            GAN.save('/media/vivo/New Volume/FERG_DB_256/model/VAEGAN_5th_MUG_8pp_real_' + str(date) + 'epochs' + str(
                ee) + '.h5')
            encoder.save('/media/vivo/New Volume/FERG_DB_256/model/VAEGAN_5th_encoder_MUG_8pp_real_' + str(
                date) + 'epochs' + str(ee) + '.h5')
            decoder.save('/media/vivo/New Volume/FERG_DB_256/model/VAEGAN_5th_decoder_MUG_8pp_real_' + str(
                date) + 'epochs' + str(ee) + '.h5')
            discriminator.save('/media/vivo/New Volume/FERG_DB_256/model/VAEGAN_5th_discriminator_MUG_8pp_real_' + str(
                date) + 'epochs' + str(ee) + '.h5')



start_time = time.time()
path = "/media/vivo/New Volume/GAN_rep_results/GAN_MUG/VAEGAN/generated/5th_8pp_" + str(date)
if os.path.isdir(path) == False:
    os.mkdir(path);


train_for_n(nb_epoch=epochs, plt_frq=500, BATCH_SIZE=batch_size)

process_time = time.time() - start_time
print("Elapsed: %s " % (process_time))
