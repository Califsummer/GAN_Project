#The code in this file implements a GAN (Generative Adversarial Network) model is for generating data that resembles real-world data

import pandas as pd
import tensorflow as tf
import numpy as np
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import LeakyReLU
from keras.initializers import RandomNormal
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

# define the discriminator model
def define_discriminator():

    init = RandomNormal(stddev=0.02)
    # define model
    model = Sequential()
    
    model.add(layers.Dense(16, use_bias=False, input_shape=(14,)))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(layers.Dense(32, use_bias=False, input_shape=(16,)))
    model.add(LeakyReLU(alpha=0.2))
    # classifier
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
  
# define the standalone generator model
def define_generator(latent_dim):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # define model
    model = Sequential()
    
    model.add(layers.Dense(16, use_bias=False, input_shape=(latent_dim,)))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(layers.Dense(32, use_bias=False, input_shape=(16,)))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(layers.Dense(64, use_bias=False, input_shape=(32,)))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(layers.Dense(14, use_bias=False, input_shape=(64,), activation='softmax'))
    return model
  
# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(generator)
    # add the discriminator
    model.add(discriminator)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model
  
def generate_real_samples(dataset, n_samples):
    
    # generate class labels
    X = dataset
    y = ones((dataset.shape[0], 1))
    return X, y
  
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input
  
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = generator.predict(x_input)
    # create class labels
    y = zeros((n_samples, 1))
    return X, y
  
# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, latent_dim, n_samples=10000):
    # prepare fake examples
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    
    X
    # save the generator model
    g_model.save('results_collapse/model_%03d.h5' % (step+1))
    
# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
    # plot loss
    pyplot.subplot(2, 1, 1)
    pyplot.plot(d1_hist, label='Discriminator-Real Loss')
    pyplot.plot(d2_hist, label='Discriminator-Fake Loss')
    pyplot.plot(g_hist, label='Generator Loss')
    pyplot.ylabel('Loss')
    pyplot.xlabel('Steps (batches per epoch * epoches)')
    pyplot.legend()
    # plot discriminator accuracy
    pyplot.subplot(2, 1, 2)
    pyplot.plot(a1_hist, label='Accuracy-Real')
    pyplot.plot(a2_hist, label='Accuracy-Fake')
    pyplot.ylabel('Accuracy')
    pyplot.xlabel('Steps (batches per epoch * epoches)')
    pyplot.legend()
    
    pyplot.show()
    # save plot to file
    pyplot.savefig('results_collapse/plot_line_plot_loss.png')
    pyplot.close()
    
# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=128):
    # calculate the number of batches per epoch
    bat_per_epo = int(dataset.shape[0] / n_batch)
    # calculate the total iterations based on batch and epoch
    n_steps = bat_per_epo * n_epochs
    # calculate the number of samples in half a batch
    half_batch = int(n_batch / 2)
    # prepare lists for storing stats each iteration
    d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
    # manually enumerate epochs
    
    for i in range(n_steps):
        # get randomly selected 'real' samples
        X_real, y_real = generate_real_samples(dataset, half_batch)
        # update discriminator model weights
        d_loss1, d_acc1 = d_model.train_on_batch(X_real, y_real)
        # generate 'fake' examples
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator model weights
        d_loss2, d_acc2 = d_model.train_on_batch(X_fake, y_fake)
        # prepare points in latent space as input for the generator
        X_gan = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))
        # update the generator via the discriminator's error
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        # summarize loss on this batch
        print('>%d, Real_Loss=%.3f, Fake_Loss=%.3f Generator_Loss=%.3f, Real_Accuracy=%d, Fake_Accuracy=%d' %
            (i+1, d_loss1, d_loss2, g_loss, int(100*d_acc1), int(100*d_acc2)))
        # record history
        d1_hist.append(d_loss1)
        d2_hist.append(d_loss2)
        g_hist.append(g_loss)
        a1_hist.append(d_acc1)
        a2_hist.append(d_acc2)
        # evaluate the model performance every 'epoch'
        if (i+1) % bat_per_epo == 0:
            summarize_performance(i, g_model, latent_dim)
    plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist)
    
    
generator = define_generator(10)

noise = tf.random.normal([1,10])
generated_data = generator(noise, training=False)

discriminator = define_discriminator()
decision = discriminator(generated_data)
print (decision)

# load data
def load_real_samples():
    df = pd.read_csv('D:\SCADA_Data\ProcessedRealData', float_precision='round_trip')
    
    # load dataset
    X = df
    y = df['isAttack']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    # select all of the examples for a given class
    selected_ix = y_train == 1
    X = X_train[selected_ix]
    # convert from ints to floats
    X = X.astype('float32')
    
    return X
  
latent_dim = 10
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)

dataset = load_real_samples()

# train model
train(generator, discriminator, gan_model, dataset, latent_dim)

xnoise = tf.random.normal([10000,latent_dim])
generated_data_res = generator(xnoise)
generated_data_res = generated_data_res.numpy()

df = pd.DataFrame(data = generated_data_res, columns = ["Source", "Destination", "Source Port", "Destination Port", "Function Code", "Protocol", "Register Value", "Reference Number", "Time since first frame in this TCP stream", "Time since previous frame in this TCP stream", "Fin", "Reset", "Stream index", "isAttack"])

# Find quantile points and label "isAttack" column based on quantile points
quantileArray = np.asarray(df['isAttack'].quantile([.5, 1]))
quantileArray

def find_nearest(array, row):
    array = np.asarray(array)
    if row['isAttack'] < 5.11332332e-09:
        return 0
    else:
        return 1
      
df['isAttack'] = df.apply(lambda row: find_nearest(quantileArray, row), axis=1)

df['isAttack'].value_counts()

df.to_csv('D:\SCADA_Data\GeneratedData', index=False)
