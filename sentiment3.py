'''
Single model may achieve LB scores at around 0.043
Don't need to be an expert of feature engineering
All you need is a GPU!!!!!!!

The code is tested on Keras 2.0.0 using Theano backend, and Python 3.5

referrence Code:https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings
'''

########################################
## import packages
########################################
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.vis_utils import plot_model





import sys

########################################
## set directories and parameters
########################################



from keras import backend as K
from keras.engine.topology import Layer
#from keras import initializations
from keras import initializers, regularizers, constraints


class Attention(Layer):
    def __init__(self, step_dim,
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
        #self.init = initializations.get('glorot_uniform')
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
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

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
        
path = '../input/data/'
EMBEDDING_FILE='../../glove.840B.300d/glove.840B.300d.txt'
TRAIN_DATA_FILE=path+'train.csv'
TEST_DATA_FILE=path+'test.csv'

MAX_SEQUENCE_LENGTH = 150
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = 300
num_dense = 256
rate_drop_lstm = 0.25
rate_drop_dense = 0.25

act = 'relu'

########################################
## index word vectors
########################################
print('Indexing word vectors')

#Glove Vectors
embeddings_index = {}
f = open(EMBEDDING_FILE)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)

########################################
## process texts in datasets
########################################
print('Processing text dataset')

#Regex to remove all Non-Alpha Numeric and space
special_character_removal=re.compile(r'[^a-z\d ]',re.IGNORECASE)

#regex to replace all numerics
replace_numbers=re.compile(r'\d+',re.IGNORECASE)

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)
    
    #Remove Special Characters
    text=special_character_removal.sub('',text)
    
    #Replace Numbers
    text=replace_numbers.sub('n',text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)


list_sentences_train = train_df["comment_text"].fillna("NA").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train_df[list_classes].values
list_sentences_test = test_df["comment_text"].fillna("NA").values


comments = []
for text in list_sentences_train:
    comments.append(text_to_wordlist(text))
    
test_comments=[]
for text in list_sentences_test:
    test_comments.append(text_to_wordlist(text))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(comments + test_comments)

sequences = tokenizer.texts_to_sequences(comments)
test_sequences = tokenizer.texts_to_sequences(test_comments)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', y.shape)

test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of test_data tensor:', test_data.shape)

########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))


########################################
## sample train/validation data
########################################
# np.random.seed(1234)
perm = np.random.permutation(len(data))
idx_train = perm[:int(len(data)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(data)*(1-VALIDATION_SPLIT)):]

data_train=data[idx_train]
labels_train=y[idx_train]
print(data_train.shape,labels_train.shape)

data_val=data[idx_val]
labels_val=y[idx_val]

print(data_val.shape,labels_val.shape)

########################################
## define the model structure
########################################
embedding_layer = Embedding(nb_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)

lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm,return_sequences=True)

comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences= embedding_layer(comment_input)
x = lstm_layer(embedded_sequences)
x = Dropout(rate_drop_dense)(x)
merged = Attention(MAX_SEQUENCE_LENGTH)(x)
merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)
preds = Dense(6, activation='sigmoid')(merged)

########################################
## train the model
########################################
model = Model(inputs=[comment_input], \
        outputs=preds)
model.compile(loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])
print(model.summary())

            # _________________________________________________________________
            # Layer (type)                 Output Shape              Param #   
            # =================================================================
            # input_1 (InputLayer)         (None, 150)               0         
            # _________________________________________________________________
            # embedding_1 (Embedding)      (None, 150, 300)          30000000  
            # _________________________________________________________________
            # lstm_1 (LSTM)                (None, 150, 300)          721200    
            # _________________________________________________________________
            # dropout_1 (Dropout)          (None, 150, 300)          0         
            # _________________________________________________________________
            # attention_1 (Attention)      (None, 300)               450       
            # _________________________________________________________________
            # dense_1 (Dense)              (None, 256)               77056     
            # _________________________________________________________________
            # dropout_2 (Dropout)          (None, 256)               0         
            # _________________________________________________________________
            # batch_normalization_1 (Batch (None, 256)               1024      
            # _________________________________________________________________
            # dense_2 (Dense)              (None, 6)                 1542      
            # =================================================================
            # Total params: 30,801,272
            # Trainable params: 800,760
            # Non-trainable params: 30,000,512
            # _________________________________________________________________
            # None



print model.summary()
plot_model(model, to_file='sentiment_model_plot.png', show_shapes=True, show_layer_names=True)


STAMP = 'simple_lstm_glove_vectors_%.2f_%.2f'%(rate_drop_lstm,rate_drop_dense)
print(STAMP)

early_stopping =EarlyStopping(monitor='val_loss', patience=5)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)


# hist = model.fit(data_train, labels_train, \
#         validation_data=(data_val, labels_val), \
#         epochs=50, batch_size=256, shuffle=True, \
#          callbacks=[early_stopping, model_checkpoint])
         
# model.load_weights(bst_model_path)

model.load_weights('simple_lstm_glove_vectors_0.25_0.25.h5')
# bst_val_score = min(hist.history['val_loss'])

#######################################
## make the submission
########################################
print('Start making the submission before fine-tuning')

y_test = model.predict([test_data], batch_size=1024, verbose=1)

print(test_data[:50])
print(y_test[:50])
'''
sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission[list_classes] = y_test

sample_submission.to_csv('%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)
'''

'''
[[9.97726977e-01 3.55345190e-01 9.72651899e-01 1.36234328e-01
  9.20513570e-01 3.30011457e-01]
 [1.02683704e-03 5.66854142e-05 2.60510133e-04 2.39637211e-05
  3.33274307e-04 1.38715870e-04]]


[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.],[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.],[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.],[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.],[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]



[[9.97726977e-01 3.55345190e-01 9.72651899e-01 1.36234328e-01
  9.20513570e-01 3.30011457e-01]
 [1.02683704e-03 5.66854142e-05 2.60510133e-04 2.39637211e-05
  3.33274307e-04 1.38715870e-04]
 [2.04553502e-03 1.70675747e-04 4.35625756e-04 8.91549425e-05
  6.85695501e-04 2.61840614e-04]
 [4.91352461e-04 1.27012790e-05 1.20294564e-04 1.74780216e-05
  1.11805763e-04 2.20156744e-05]
 [4.42874804e-03 1.29068270e-04 6.14125049e-04 9.50055401e-05
  8.02407041e-04 1.77977650e-04]
 [3.33683507e-04 2.56531384e-05 8.03477815e-05 3.81883037e-05
  2.02600655e-04 6.74169642e-05]
 [1.57878140e-03 9.40056998e-06 1.53176312e-04 8.62383058e-06
  2.42249473e-04 1.89233451e-05]
 [2.45815516e-01 4.58688999e-04 1.21936444e-02 1.26102837e-04
  3.56645845e-02 5.47127158e-04]
 [6.73368648e-02 1.42904637e-05 2.31599202e-03 1.00580346e-05
  1.01204468e-02 1.90319115e-04]
 [8.91864067e-04 2.33772498e-05 1.56531692e-04 3.05063586e-05
  1.94393418e-04 5.34378996e-05]
 [3.12245518e-01 4.89355472e-04 7.49237463e-02 1.28940592e-04
  1.43747870e-02 4.68472746e-04]
 [1.01953633e-01 9.12855394e-05 5.97724412e-03 6.91065434e-05
  7.64344679e-03 1.08171732e-03]
 [1.61243021e-03 5.96867676e-06 1.27419131e-04 3.09002485e-06
  1.56591559e-04 1.89723305e-05]
 [1.16520889e-04 7.92745425e-07 1.94080931e-05 8.93122873e-08
  1.07672295e-05 4.06071877e-06]
 [1.53347733e-04 3.79522521e-06 5.77519604e-05 1.08315953e-06
  2.49004388e-05 6.46533044e-06]
 [1.02432258e-03 2.90721273e-05 1.86194331e-04 1.65998772e-05
  2.27697179e-04 6.14325254e-05]
 [4.06514015e-03 1.53640445e-04 7.46057776e-04 7.46732767e-05
  1.36637094e-03 3.29379342e-04]
 [3.26533173e-03 3.17786325e-05 4.22653276e-04 1.76071080e-05
  4.65551304e-04 9.20065213e-05]
 [4.37616574e-04 3.89282195e-06 1.07801738e-04 2.42539454e-06
  4.29752217e-05 9.04626449e-06]
 [1.02603808e-03 8.97609571e-05 2.50850542e-04 3.54863005e-05
  3.31937452e-04 9.49904061e-05]
 [3.41743999e-03 2.06865589e-04 5.46837808e-04 4.79808776e-04
  4.90694831e-04 1.31592256e-04]
 [7.52376974e-01 3.28616127e-02 8.68195072e-02 8.32508206e-02
  1.35793686e-01 4.93849277e-01]
 [1.44307181e-01 2.36697495e-03 6.21287292e-03 1.14631383e-02
  1.93335246e-02 5.43296486e-02]
 [1.54658919e-04 1.73409990e-06 4.56365597e-05 6.71492558e-07
  2.70464661e-05 6.55471104e-06]
 [1.32436663e-01 4.95312328e-04 1.74820423e-02 1.59789692e-04
  8.51882342e-03 3.33685311e-03]
 [9.26452485e-05 9.39455958e-07 1.33933690e-05 9.34081299e-07
  1.98585003e-05 2.09107270e-06]
 [8.22297763e-04 4.26004372e-05 1.86495337e-04 5.79159241e-05
  2.14947155e-04 1.17403324e-04]
 [7.31679099e-03 7.61733463e-05 7.12904788e-04 3.79261983e-05
  1.58639031e-03 2.16077111e-04]
 [1.26049414e-01 1.05083978e-03 4.41953260e-03 9.46349115e-04
  1.51700452e-02 1.38661683e-01]
 [6.58207980e-04 6.54360065e-06 1.45526094e-04 8.14377654e-06
  1.08846973e-04 2.24618780e-05]
 [1.13640109e-03 4.43528297e-05 2.52848346e-04 2.46102518e-05
  2.85555521e-04 7.36119764e-05]
 [1.65758689e-03 1.21024925e-04 2.93649617e-04 1.28171610e-04
  7.89064739e-04 2.95414182e-04]
 [3.21666768e-04 1.30657218e-05 7.48280290e-05 1.04826922e-05
  7.33371780e-05 3.06081783e-05]
 [8.91320014e-05 1.50331425e-06 4.58867908e-05 1.48047320e-07
  1.24647186e-05 4.89853619e-06]
 [4.88670217e-03 3.26645590e-04 8.74916906e-04 1.81238458e-04
  1.70527596e-03 4.28201107e-04]
 [3.75049165e-03 4.47211460e-05 4.87408077e-04 4.23340425e-05
  9.47914785e-04 1.30257322e-04]
 [6.15561439e-05 4.36875825e-06 2.14998272e-05 2.11548127e-06
  1.68848055e-05 1.00036514e-05]
 [7.20466999e-03 2.81903951e-04 6.57620432e-04 7.48123799e-04
  1.93996762e-03 3.37963633e-04]
 [9.00655091e-01 5.91940805e-03 8.44111323e-01 7.90975246e-05
  5.07422626e-01 3.19456644e-02]
 [1.79849580e-04 7.20052583e-07 4.01595898e-05 2.92942531e-07
  8.56155202e-06 2.52212358e-05]
 [8.35740194e-03 2.20201196e-04 9.11208335e-04 9.67610031e-05
  1.91343972e-03 2.57353648e-04]
 [5.16800303e-03 1.93594155e-06 2.54299433e-04 1.39184829e-06
  2.98306637e-04 1.28080392e-05]
 [1.41939978e-04 1.60644158e-05 6.55616022e-05 6.53419693e-06
  1.32283065e-04 2.53836752e-05]
 [2.75766128e-04 3.03322906e-07 6.92065223e-05 6.32555910e-08
  1.24755779e-05 1.01304840e-06]
 [1.54808059e-03 5.78068921e-07 1.48315856e-04 8.76925998e-09
  2.66976695e-05 1.75206799e-06]
 [9.50604590e-05 3.09891475e-05 4.73718646e-05 1.76407557e-05
  3.22358683e-05 9.23146363e-05]
 [1.55805759e-02 2.71272384e-05 1.16095482e-03 1.58150451e-05
  1.34630338e-03 8.49577118e-05]
 [7.84245844e-04 2.82701803e-05 2.23575116e-04 7.33767592e-06
  1.44757170e-04 8.52799567e-05]
 [9.94760454e-01 3.64682764e-01 9.21685100e-01 6.94125099e-03
  8.72205913e-01 7.79416740e-01]
 [1.85152981e-03 5.71892269e-05 2.87028903e-04 2.96468661e-05
  4.56112582e-04 3.47128807e-04]]
'''