import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import time
import tensorflow as tf
import numpy as np # linear algebra
import random
import os 
os.environ['PYTHONHASHSEED'] = '11'
np.random.seed(22)
random.seed(33)
tf.set_random_seed(44)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
import gc

from keras import initializers, regularizers, constraints
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding, CuDNNGRU, CuDNNLSTM
from keras.layers import Bidirectional, GlobalMaxPooling1D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
from keras.callbacks import Callback
from keras.models import clone_model
import keras.backend as K

max_features = 90000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 50
t0 = time.time()

train_df = pd.read_csv("data/train.csv")
print("Train shape : ",train_df.shape)

# fill up the missing values
train_X = train_df["question_text"].fillna("_na_").values


train_X = train_df['question_text']
train_X = train_X.tolist()

qid_train = train_df['qid']
qid_train = qid_train.tolist()

# Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features,
                     filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'’“”')


tokenizer.fit_on_texts(train_X)
train_X = tokenizer.texts_to_sequences(train_X)

# Pad the sentences 
trunc = 'pre'
train_X = pad_sequences(train_X, maxlen=maxlen, truncating=trunc)

# Get the target values
train_y = train_df['target'].values

test_X = train_X[1000000:]
train_X = train_X[:1000000]
test_y = train_y[1000000:]
train_y = train_y[:1000000]

from scipy.io import loadmat
embedding_matrix = loadmat('data/embedding_matrix.mat')["embedding_matrix"]

print(embedding_matrix.shape)

from keras.engine.topology import Layer

class Attention(Layer):
    def __init__(self, step_dim=50,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim

from keras.layers import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D

def create_rnn_model(rnn, lstm, maxlen, embedding, max_features, embed_size,
                     rnn_dim=64, dense1_dim=100, dense2_dim=50,
                     embed_trainable=False, seed=123):
    inp = Input(shape=(maxlen,), dtype='int32')
    x = Embedding(max_features, embed_size, weights=[embedding],
                  trainable=embed_trainable)(inp)
    x = Bidirectional(lstm(rnn_dim, kernel_initializer=glorot_uniform(seed=seed),
                           return_sequences=True))(x)
    
    
    x = Attention(maxlen)(x)
    merged = Dense(dense1_dim, activation='relu')(x)
    merged = BatchNormalization()(merged)
    preds = Dense(1, activation='sigmoid')(merged)
    model = Model(inputs=[inp], \
        outputs=preds)
    model.compile(loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model

k = 4
num_val_samples = 200000 
num_epochs = 2
all_scores = []

gc.collect()

def f1_best(y_val, pred_val):
    return metrics.f1_score(y_val,pred_val)

embed_ids = [list(range(300)), list(range(300, 600)),
             list(range(600, 900)), list(range(900, 1200))]
embed_ids_dict = {1: [embed_ids[0], embed_ids[1], embed_ids[2], embed_ids[3]],
                  2: [embed_ids[0] + embed_ids[1],
                      embed_ids[0] + embed_ids[2],
                      embed_ids[0] + embed_ids[3],
                      embed_ids[1] + embed_ids[2],
                      embed_ids[1] + embed_ids[3],
                      embed_ids[2] + embed_ids[3]],
                  3: [embed_ids[0] + embed_ids[1] + embed_ids[2],
                      embed_ids[0] + embed_ids[1] + embed_ids[3],
                      embed_ids[0] + embed_ids[2] + embed_ids[3],
                      embed_ids[1] + embed_ids[2] + embed_ids[3]],
                  4: [embed_ids[0] + embed_ids[1] + embed_ids[2] + embed_ids[3]]}
embed_ids_lst = embed_ids_dict[2]
embed_size = 1200

rnn = CuDNNGRU
lstm = CuDNNLSTM
embed_trainable = False

n_models = 1
#epochs = 7
batch_size = 128
dense1_dim = rnn_dim = 128
dense2_dim = 2 * rnn_dim

ema_n = int(len(train_y) / batch_size / 10)
decay = 0.9
scores = []
seed = 101 + 11 * 1
    
for i in range(n_models):
    t1 = time.time()
    seed = 101 + 11 * i
    cols_in_use = embed_ids_lst[i % len(embed_ids_lst)]
    
    model = create_rnn_model(rnn, lstm, maxlen, embedding_matrix,
                             max_features, embed_size,
                             rnn_dim=rnn_dim,
                             dense1_dim=dense1_dim,
                             dense2_dim=dense2_dim,
                             embed_trainable=embed_trainable,
                             seed=seed)
    model.summary()

    
    #cross_validation 
    for i in range(k):
        print(f'Processing fold # {i}')
        val_data = train_X[i * num_val_samples: (i+1) * num_val_samples]
        val_targets = train_y[i * num_val_samples: (i+1) * num_val_samples]
    
        partial_train_data = np.concatenate(
                                [train_X[:i * num_val_samples],
                                train_X[(i+1) * num_val_samples:]],
                                axis=0)
        partial_train_targets = np.concatenate(
                                [train_y[:i * num_val_samples],
                                train_y[(i+1)*num_val_samples:]],
                                axis=0)
        res = model.fit(x=partial_train_data,
                        y=partial_train_targets,
                        batch_size=batch_size,
                        epochs = num_epochs,
                        verbose=1,
                        validation_data = (val_data,val_targets))
        print("Scores: ", res)