import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import os
import time
pd.set_option('display.float_format',lambda x : '%.1f' % x)

tf.enable_eager_execution()

#Data preprocessing
df = pd.read_csv('D:\SCADA_Data\scadaData.csv')
df1 = df.iloc[:,[0,1,2,3,5,6,7,8,9,12]]
df1.columns = ['SourceIP','DestIP','SourcePort','DestPort','TransID','FuncCode','RefNo','WriteData','RespData','Alarm']
df1[['column_pad_1', 'column_pad_2', 'column_pad_3', 'column_pad_4', 'column_pad_5', 'column_pad_6']] = pd.DataFrame([[0, 0, 0, 0, 0, 0]], index=df.index)
traindata = df1.to_numpy()
traindata_rs = traindata.reshape([traindata.shape[0], 4, 4, 1])
traindata_rs = np.float32(traindata_rs)
  
  
#Define generator model
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*4*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((4, 4, 256)))
    assert model.output_shape == (None, 4, 4, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 4, 4, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 4, 4, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 4, 4, 1)
    
    model.add(layers.Reshape((1, 16)))

    return model
  
  
generator = make_generator_model()


#See if the generator works well before training it
noise = tf.random.normal([1,100])
generated_data = generator(noise, training=False)
generated_data = generated_data.numpy()


#Define discriminator model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[4, 4, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
  
#Reshape the generated data to fit in the input layer of discriminator
generated_data_rd = generated_data.reshape(1, 4, 4, 1)

#Discriminate using discriminator to see if the it works well
discriminator = make_discriminator_model()
decision = discriminator(generated_data_rd)
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

EPOCHS = 150
noise_dim = 100
num_examples_to_generate = 16

#Generate random seed to feed into generator model
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# This annotation of `tf.function` causes the function to be "compiled".
@tf.function
def train_step(data):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_data = generator(noise, training=True)
        #generated_data = generated_data.numpy()
        generated_data_rd = tf.reshape(generated_data, [BATCH_SIZE, 4, 4, 1])

        real_output = discriminator(data, training=True)
        fake_output = discriminator(generated_data_rd, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
 
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for data_batch in dataset:
            train_step(data_batch)

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
    
    
def generate_and_save_data(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    
    for i in range(predictions.shape[0]):
        print(predictions[i])
        
        
train(train_dataset, EPOCHS)
  
  
#After training, generate new noises and feed into generator
xnoise = tf.random.normal([1,100])
generated_datax = generator(xnoise, training=False)
