# Another implementation of GAN

import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import os
import time
pd.set_option("precision", 20)

pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_rows', 1000)

df = pd.read_csv('D:\SCADA_Data\ProcessedRealData', float_precision='round_trip')

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(16, use_bias=False, input_shape=(10,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Dense(32, use_bias=False, input_shape=(16,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Dense(64, use_bias=False, input_shape=(32,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Dense(14, use_bias=False, input_shape=(64,), activation='softmax'))

    return model
  
generator = make_generator_model()

noise = tf.random.normal([1,10])
generated_data = generator(noise, training=False)


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, use_bias=False, input_shape=[14]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.8))

    model.add(layers.Dense(128, use_bias=False, input_shape=(256,)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.8))
              
    model.add(layers.Dense(64, use_bias=False, input_shape=(128,)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.8))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
  
  
# An alternative discriminator  
def make_discriminator_model(num_row): lstm_out = 200 batch_size = 32

    model = tf.keras.Sequential()
    model.add(layers.LSTM(lstm_out, input_shape =(14,num_row), dropout = 0.2))
    model.add(layers.Dense(2,activation='sigmoid'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

    return model
  
  
discriminator = make_discriminator_model()
decision = discriminator(generated_data)
print (decision)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
  
def generator_loss(fake_output):
  return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 50
noise_dim = 10
num_examples_to_generate = 16

# Generate random seed to be the input of generator
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(data, gen_hist, disc_hist):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_data = generator(noise, training=True)
        
        #discriminator = make_discriminator_model(generated_data.shape[0])
        real_output = discriminator(data, training=True)
        fake_output = discriminator(generated_data, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        
        

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    
    gen_hist.append(gen_loss)
    disc_hist.append(disc_loss)
    
    
# create a line plot of loss and accuracy for the model and save to file
def plot_history(gen_hist, disc_hist):
    
    # plot loss
    pyplot.subplot(2, 1, 1)
    plt.plot(gen_hist, label='gen_loss')
    plt.plot(disc_hist, label='disc_loss')
    
    
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    
    # plot discriminator accuracy
    pyplot.subplot(2, 1, 2)
    pyplot.plot(a1_hist, label='acc-real')
    pyplot.plot(a2_hist, label='acc-fake')
    pyplot.legend()
    
    # save plot to file
    plt.savefig('results_baseline/plot_line_plot_loss.png')
    plt.show()
    
    
def train(dataset, epochs):
    gen_hist, disc_hist = list(), list()
    for epoch in range(epochs):
        start = time.time()

        for data_batch in dataset:
            train_step(data_batch, gen_hist, disc_hist)

        # Produce data as we go
        generate_and_save_data(generator,
                               epoch + 1,
                               seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_data(generator,
                             epochs,
                             seed)
    plot_history(gen_hist, disc_hist)
    
    
def generate_and_save_data(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    
    for i in range(predictions.shape[0]):
        print(predictions[i])
        
        
BUFFER_SIZE = 30000
BATCH_SIZE = 2048
train_dataset = tf.data.Dataset.from_tensor_slices(df).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

train(train_dataset, EPOCHS)

xnoise = tf.random.normal([10000,10])
generated_data_res = generator(xnoise)

generated_data_res = generated_data_res.numpy()

df = pd.DataFrame(data = generated_data_res, columns = ["Source", "Destination", "Source Port", "Destination Port", "Function Code", "Protocol", "Register Value", "Reference Number", "Time since first frame in this TCP stream", "Time since previous frame in this TCP stream", "Fin", "Reset", "Stream index", "isAttack"])

# Find quantile points and label "isAttack" column based on quantile points
quantileArray = np.asarray(df['isAttack'].quantile([.5, 1]))
def find_nearest(array, row):
    array = np.asarray(array)
    if row['isAttack'] < 4.44551201e-10:
        return 0
    else:
        return 1

df['isAttack'] = df.apply(lambda row: find_nearest(quantileArray, row), axis=1)

df['isAttack'].unique()
df['isAttack'].value_counts()
df.to_csv('D:\SCADA_Data\GeneratedDataHAHA', index=False)
