"Implementation of PPRL-VGAN for FERG dataset"
import os, random

os.environ["KERAS_BACKEND"] = "tensorflow"
import scipy.io
import cPickle, random, sys, keras
from PIL import Image
from keras.layers import Conv2D
import h5py
import numpy as np
from keras.layers import Input, Lambda
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.normalization import *
from keras.optimizers import *
from keras import initializers
import matplotlib.pyplot as plt
from keras.models import Model
from tqdm import tqdm
from skimage.measure import compare_ssim as ssim
import time
import scipy
from functools import partial

normal = partial(initializers.normal, scale=.02)


################################## load and preprocess the dataset #####################################
batch_size  = 256
num_ep      = 7
num_pp      = 6
epochs      = 400
img_rows, img_cols = 64, 64
clipvalue   = 20
noise_dim   = 10
c_dim       = num_pp
n_dim       = 10
date        = 2018
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


print('label 2 shape', label2.shape)
x     =  f['images']
x     = np.asarray(x);
x     = np.transpose(x, [3,2,1,0])
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
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('label 1 train', y_train1.shape)
print('label 1 test', y_test1.shape)
x_train = (x_train- 127.5)/127.5
x_test  = (x_test- 127.5)/127.5
x_train = x_train.astype('float16')
x_test  = x_test.astype('float16')

y_train1 = keras.utils.to_categorical(y_train1, num_pp)
y_test1  = keras.utils.to_categorical(y_test1, num_pp)
y_train2 = keras.utils.to_categorical(y_train2, num_ep)
y_test2  = keras.utils.to_categorical(y_test2, num_ep)


x_ori = (x - 127.5) / 127.5

epsilon_std = 1
def sampling_np(args):
    z_mean, z_log_var = args
    epsilon = np.random.normal(loc=0., scale=epsilon_std, size=(z_mean.shape[0], z_dim), )
    return z_mean + np.exp(z_log_var / 2) * epsilon


def generate_dataset():
    ## save to numpyz###############
    c = np.random.randint(num_pp, size=x_train.shape[0])
    c_train = keras.utils.to_categorical(c, num_pp)
    c = np.random.randint(num_pp, size=x_test.shape[0])
    c_test = keras.utils.to_categorical(c, num_pp)
    [z_train, mean_var_train] = encoder.predict(x_train)
    [z_test, mean_var_test] = encoder.predict(x_test)
    np.savez('/media/vivo/New Volume/FERG_DB_256/new_rep/FERG_color' + str(
        date) + '_fi_512_VAE_GAN_labelfull_para_opt.npz', z_train, y_train1, y_train2, c_train, z_test, y_test1,
             y_test2, c_test)

rep_field = 5
opt  = RMSprop(lr=0.0003, decay=1e-6)
dopt = RMSprop(lr=0.0003, decay=1e-6)

# KL divergence
def KL_loss(y_true, y_pred):
    z_mean = y_pred[:, 0:z_dim]
    z_log_var = y_pred[:, z_dim:2 * z_dim]
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(kl_loss)

# function for sampling a latent vector based on given mean and standard deviation
def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], z_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_sigma / 2) * epsilon

def model_encoder(z_dim, input_shape, units=512, dropout=0.3, rep_field=None):
    k = rep_field
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

def model_decoder(z_dim, c_dim, rep_field=None):
    k = rep_field
    x = Input(shape=(z_dim,))
    auxiliary_c = Input(shape=(c_dim,), name='aux_input_c')
    # auxiliary_z = Input(shape=(n_dim,), name='aux_input_z')
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
k = rep_field
input_shape = (img_rows, img_cols, 3)
d_input = Input(input_shape)

x = Conv2D(32, (k, k), strides=(2, 2), padding='same', name='id_conv1')(d_input)
# x = BatchNormalization(momentum=0.9)(x)
x = LeakyReLU(0.2)(x)

x = Conv2D(64, (k, k), strides=(2, 2), padding='same', name='id_conv2')(x)
# x = BatchNormalization(momentum=0.9)(x)
x = LeakyReLU(0.2)(x)

x = Conv2D(128, (k, k), strides=(2, 2), padding='same', name='id_conv3')(x)
# x = BatchNormalization(momentum=0.9)(x)
x = LeakyReLU(0.2)(x)

x = Conv2D(256, (k, k), strides=(2, 2), padding='same', name='id_conv4')(x)
# x = BatchNormalization(momentum=0.9)(x)
x = LeakyReLU(0.2)(x)

x = Flatten()(x)
x = Dense(256, name='ds')(x)
x = LeakyReLU(0.2)(x)
x = Dropout(0.5)(x)
output_binary = Dense(1, activation='sigmoid', name='bin_real')(x)
output_identity = Dense(num_pp, activation='softmax', name='id_real')(x)
output_expression = Dense(num_ep, activation='softmax', name='exp_real')(x)

discriminator = Model(d_input, [output_binary, output_identity, output_expression])
discriminator.compile(loss=['binary_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
                      loss_weights=[1 / 4.0, 1 / 2.0, 1 / 4.0], optimizer=dopt, metrics=['accuracy'])
discriminator.summary()
print (discriminator.metrics_names)

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

make_trainable(discriminator, False)
discriminator.trainable = False

###### Build GAN model ######

ee = 1000
input_img = Input(shape=(img_rows, img_cols, 3))
z_dim = 128
units = 256
auxiliary_c = Input(shape=(c_dim,), name='aux_input_c')
Img = Input((img_rows, img_cols, 3))
encoder = model_encoder(z_dim=z_dim, input_shape=(img_rows, img_cols, 3), units=units, dropout=0.3,
                        rep_field=rep_field)
# encoder.load_weights('')
encoder.compile(loss='binary_crossentropy', optimizer=opt)
# encoder.summary()

decoder = model_decoder(z_dim=z_dim, c_dim=c_dim, rep_field=rep_field)
decoder.compile(loss='binary_crossentropy', optimizer=opt)

[z, mean_var] = encoder(Img)
xpred = decoder([z, auxiliary_c])
output_binary, output_identity, output_expression = discriminator(xpred)
GAN = Model([Img, auxiliary_c], [mean_var, output_binary, output_identity, output_expression])
GAN.compile(loss=[KL_loss, 'binary_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
            loss_weights=[0.002, 0.108, 0.6, 0.29], optimizer=opt, metrics=['accuracy'])
GAN.summary()
print (GAN.metrics_names)

###
# generate_dataset()
###


def plotGeneratedImages(epoch, idx=0, examples=10, dim=(10, 10), figsize=(10, 10)):

    n = num_pp  # how many digits we will display
    pp_avg = 4500
    plt.figure(figsize=(16, 4))
    sample = x_ori[idx:idx + n, :, :, :]
    # sample = np.repeat(sample[np.newaxis, :, :, : ], n, axis=0)
    c = np.asarray([0, 1, 2, 3, 4, 5])
    c = keras.utils.to_categorical(c, num_pp)

    [z, mean_var] = encoder.predict(sample)
    generated_images = decoder.predict([z, c])
    [bin, identity, expression] = discriminator.predict(generated_images)
    print("identity prediction", np.argmax(identity, axis=1))

    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        ori = sample[i].reshape(img_rows, img_cols, 3)
        ori = np.uint8(ori * 127.5 + 127.5)
        plt.imshow(ori)
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax  = plt.subplot(2, n, i + 1 + n)
        rec = generated_images[i].reshape(img_rows, img_cols, 3)
        rec = np.uint8(rec * 127.5 + 127.5)
        plt.imshow(rec)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(path + '/_' + str(date) + '_generated_image_epoch_%d.tif' % epoch)
    plt.close()

def val_test():
    num_test = x_test.shape[0]
    c = np.random.randint(num_pp, size=x_test.shape[0])
    gt_id = c
    print('gt_id', gt_id.shape)
    c = keras.utils.to_categorical(c, num_pp)
    [z, mean_var] = encoder.predict(x_test)
    x_pred = decoder.predict([z, c])
    bin_val, id_val, ep_val = discriminator.predict(x_pred)
    bin_val = np.argmax(bin_val, axis=1)
    id_val = np.argmax(id_val, axis=1)
    ep_val = np.argmax(ep_val, axis=1)
    bin_gt = np.ones((x_test.shape[0], 1))
    count0 = 0
    count1 = 0
    count2 = 0

    for i in xrange(0, num_test):
        if int(bin_val[i]) == int(bin_gt[i, :]):
            # print(int(bin_val[i]))
            # print(int(bin_gt[i,:]))
            count0 += 1.0
        if int(id_val[i]) == int(gt_id[i]):
            # print(id_val[i])
            # print(y_test1_ori[:,i])
            count1 += 1.0
        if int(ep_val[i]) == int(y_test2_ori[i, :]):
            # print('val',int(ep_val[i]))
            # print('gt',int(y_test2[:,i]))
            count2 += 1.0
    print(count0)
    print(count1)
    print(count2)
    print('bin acc=', count0 / num_test)
    print('id acc=', count1 / num_test)
    print('act acc=', count2 / num_test)
    val_bin_acc.append(count0 / num_test)
    val_id_acc.append(count1 / num_test)
    val_ep_acc.append(count2 / num_test)
    return val_bin_acc, val_id_acc, val_ep_acc

val_bin_acc = []
val_id_acc = []
val_ep_acc = []
Dist_loss = []
Dist_bin_loss = []
Dist_id_loss = []
Dist_exp_loss = []
Dist_bin_acc = []
Dist_id_acc = []
Dist_exp_acc = []
GAN_loss = []
GAN_KL_loss = []
GAN_bin_loss = []
GAN_id_loss = []
GAN_exp_loss = []
GAN_bin_acc = []
GAN_id_acc = []
GAN_exp_acc = []


def train_for_n(nb_epoch=50000, plt_frq=25, BATCH_SIZE=256):
    batchCount = x_train.shape[0] / BATCH_SIZE
    for ee in xrange(1, nb_epoch + 1):
        print '-' * 15, 'Epoch %d' % ee, '-' * 15
        plotGeneratedImages(epoch=ee, idx=75)
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

            y0_dist_real = np.random.uniform(0.9, 1.0, size=[BATCH_SIZE, 1])
            y0_dist_fake = np.random.uniform(0, 0.1, size=[BATCH_SIZE, 1])

            make_trainable(discriminator, True)
            discriminator.trainable = True
            d_loss_real = discriminator.train_on_batch(image_batch, [y0_dist_real, y1_batch, y2_batch])
            d_loss_fake = discriminator.train_on_batch(generated_images, [y0_dist_fake, c, y2_batch])
            d_loss = []
            for i in xrange(0, 7):
                d_loss.append((d_loss_real[i] + d_loss_fake[i]) / 2.0)

            Dist_loss.append(d_loss[0])
            Dist_bin_loss.append(d_loss[1])
            Dist_id_loss.append(d_loss[2])
            Dist_exp_loss.append(d_loss[3])
            Dist_bin_acc.append(d_loss[4])
            Dist_id_acc.append(d_loss[5])
            Dist_exp_acc.append(d_loss[6])
            print('discriminator loss', d_loss[0])
            # print('discriminator binary acc', d_loss_real[1])
            print('discriminator identification real acc', d_loss_real[5])
            print('discriminator identification fake acc', d_loss_fake[5])
            print('discriminator expression real acc', d_loss_real[6])
            print('discriminator expression fake acc', d_loss_fake[6])
            # print('discriminator action recognition acc', d_loss_real[6])



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

                y0_batch = np.ones((BATCH_SIZE, 1))
                g_loss = GAN.train_on_batch([image_batch, c], [mean_var_ref, y0_batch, c, y2_batch])
                GAN_loss.append(g_loss[0])
                GAN_KL_loss.append(g_loss[1])
                GAN_bin_loss.append(g_loss[2])
                GAN_id_loss.append(g_loss[3])
                GAN_exp_loss.append(g_loss[4])
                GAN_bin_acc.append(g_loss[6])
                GAN_id_acc.append(g_loss[7])
                GAN_exp_acc.append(g_loss[8])
                print('GAN loss', g_loss[0])
                print('GAN KL loss', g_loss[1])

        if ee % 15 == 0:
            GAN.save('GAN_weights.h5')
            encoder.save('encoder_weights.h5')
            decoder.save('decoder_weights.h5')
            discriminator.save('disc_weights.h5')

            # np.savez('./dataset/stats-' + str(date) + '_' + str(bc) + '-blabla.npz', Dist_loss,
            #          Dist_bin_loss, Dist_id_loss, Dist_exp_loss, Dist_bin_acc, Dist_id_acc, Dist_exp_acc, GAN_loss,
            #          GAN_KL_loss, GAN_bin_loss, GAN_id_loss, GAN_exp_loss, GAN_bin_acc, GAN_id_acc, GAN_exp_acc,
            #          val_bin_acc, val_id_acc, val_ep_acc)

start_time = time.time()
path = "/your_path_" + '_' + str(date)

if os.path.isdir(path) == False:
    os.mkdir(path);

train_for_n(nb_epoch=epochs, plt_frq=500, BATCH_SIZE=batch_size)
process_time = time.time() - start_time
print("Elapsed: %s " % (process_time))
