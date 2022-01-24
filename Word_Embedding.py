# As the data acquired from SCADA system include string, integer and category data type, we cannot put them into machine learning models directly
# The code in this file uses word embedding to convert the data into a form that can fit the input layer in GAN
import pandas as pd
import tensorflow as tf
import numpy as np
import re
from keras.models import Input, Model
from keras.layers import Dense
import os
import time
import sys
import matplotlib.pyplot as plt
from scipy import sparse
from tqdm import tqdm
np.set_printoptions(threshold=sys.maxsize)

df = pd.read_csv('D:\SCADA_Data\ProcessedRealData', float_precision='round_trip')

x = df.to_string(header=False,
                  index=False,
                  index_names=False).split('\n')
attackStrings = [','.join(ele.split()) for ele in x]
for s in attackStrings:
    print(s,'\n')

# Defining the window for context
window = 2

# Creating a placeholder for the scanning of the word list
attack_lists = []
all_attack = []

# Creating a context dictionary
for i, attack in enumerate(attackStrings):
    for w in range(window):
        # Getting the context that is ahead by *window* words
        if i + 1 + w < len(attackStrings): 
            attack_lists.append([attack] + [attackStrings[(i + 1 + w)]])
        # Getting the context that is behind by *window* words    
        if i - w - 1 >= 0:
            attack_lists.append([attack] + [attackStrings[(i - w - 1)]])
            
def create_unique_word_dict(attackStrings:list) -> dict:
    """
    A method that creates a dictionary where the keys are unique words
    and key values are indices
    """
    # Getting all the unique words from our text and sorting them alphabetically
    attacks = list(set(attackStrings))
    print (attacks)
    #words.sort()

    # Creating the dictionary for the unique words
    unique_word_dict = {}
    for i, word in enumerate(attacks):
        unique_word_dict.update({
            word: i
        })

    return unique_word_dict 
  
unique_word_dict = create_unique_word_dict(attackStrings)

# Defining the number of features (unique words)
n_attacks = len(unique_word_dict)

# Getting all the unique words 
attacks = list(unique_word_dict.keys())

# Creating the X and Y matrices using one hot encoding
X = []
Y = []

for i, attack_list in tqdm(enumerate(attack_lists)):
    # Getting the indices
    main_word_index = unique_word_dict.get(attack_list[0])
    context_word_index = unique_word_dict.get(attack_list[1])

    # Creating the placeholders   
    X_row = np.zeros(n_attacks)
    Y_row = np.zeros(n_attacks)

    # One hot encoding the main word
    X_row[main_word_index] = 1

    # One hot encoding the Y matrix words 
    Y_row[context_word_index] = 1

    # Appending to the main matrices
    X.append(X_row)
    Y.append(Y_row)

# Converting the matrices into an array
X = np.asarray(X)
Y = np.asarray(Y)

# Defining the size of the embedding
embed_size = 2

# Defining the neural network
inp = Input(shape=(X.shape[1],))
x = Dense(units=embed_size, activation='tanh')(inp)
x = Dense(units=Y.shape[1], activation='softmax')(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

# Optimizing the network weights
model.fit(
    x=X, 
    y=Y, 
    batch_size=256,
    epochs=400
    )

# Obtaining the weights from the neural network. 
# These are the so called word embeddings

# The input layer 
weights = model.get_weights()[0]

# Creating a dictionary to store the embeddings in. The key is a unique word and 
# the value is the numeric vector
embedding_dict = {}
for attack in attacks: 
    embedding_dict.update({
        attack: weights[unique_word_dict.get(attack)]
        })
    
    
plt.figure(figsize=(10, 10))
for attack in list(unique_word_dict.keys()):
    coord = embedding_dict.get(attack)
    plt.scatter(coord[0], coord[1])
    plt.annotate(attack, (coord[0], coord[1]))
