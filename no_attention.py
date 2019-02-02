
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.normalization import BatchNormalization

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.vis_utils import plot_model


path = '../input/data/'
EMBEDDING_FILE = '../../glove.840B.300d/glove.840B.300d.txt'
TRAIN_DATA_FILE = path+'train.csv'
TEST_DATA_FILE = path+'test.csv'

MAX_SEQUENCE_LENGTH = 150
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = 300
num_dense = 64
rate_drop_lstm = 0.25
rate_drop_dense = 0.25

act = 'relu'





def softmax(x, axis=1):
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')





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

# print(len(embeddings_index))
# print train_df.shape, test_df.shape
# print(train_df.head())
# print(test_df.head())





special_character_removal=re.compile(r'[^a-z\d ]',re.IGNORECASE)
replace_numbers=re.compile(r'\d+',re.IGNORECASE)

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):

    text = text.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)    
    text=special_character_removal.sub('',text)
    text=replace_numbers.sub('n',text)

    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
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

print(len(comments),len(test_comments))    
print(comments[:5])
print(test_comments[:5])





tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(comments + test_comments)

sequences = tokenizer.texts_to_sequences(comments)
test_sequences = tokenizer.texts_to_sequences(test_comments)

print(len(sequences), len(test_sequences))
print(sequences[:5])
print(test_sequences[:5])

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', y.shape)

test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of test_data tensor:', test_data.shape)



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





perm = np.random.permutation(len(data))
idx_train = perm[:int(len(data)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(data)*(1-VALIDATION_SPLIT)):]

data_train=data[idx_train]
labels_train=y[idx_train]
print(data_train.shape,labels_train.shape)

data_val=data[idx_val]
labels_val=y[idx_val]

print(data_val.shape,labels_val.shape)





densor1 = Dense(32, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights') 
dotor = Dot(axes = 1)

def one_step_attention(a):
    e = densor1(a)
    energies = densor2(e)
    alphas = activator(energies)
    context = dotor([alphas,a])
    return context

n_a = 64
embedding_layer = Embedding(nb_words,EMBEDDING_DIM, 
							weights=[embedding_matrix],
					        input_length=MAX_SEQUENCE_LENGTH,
					        trainable=False)

bi_lstm_layer = Bidirectional(LSTM(n_a, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=False))
bi_cudnn_lstm_layer = Bidirectional(CuDNNLSTM(n_a, return_sequences=False))

lstm_layer = LSTM(n_a, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=False)
cudnn_lstm_layer = CuDNNLSTM(n_a, return_sequences=False)

comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences= embedding_layer(comment_input)
# x = bi_lstm_layer(embedded_sequences)
# x = lstm_layer(embedded_sequences)
x = bi_lstm_layer(embedded_sequences)
x = Dropout(rate_drop_dense)(x)
# context = one_step_attention(x)
# context = Flatten()(context)
merged = Dense(num_dense, activation=act)(x)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)
preds = Dense(6, activation='sigmoid')(merged)
model = Model(inputs=[comment_input], \
        outputs=preds)
model.compile(loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])
print(model.summary())








plot_model(model, to_file='sentiment_model_plot_diff_without_attention_bi_lstm.png', show_shapes=True, show_layer_names=True)


STAMP = 'simple_lstm_glove_vectors_diff_without_attention_bi_lstm_%.2f_%.2f'%(rate_drop_lstm,rate_drop_dense)
print(STAMP)

early_stopping =EarlyStopping(monitor='val_loss', patience=5)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)


hist = model.fit(data_train, labels_train, \
        validation_data=(data_val, labels_val), \
        epochs=50, batch_size=256, shuffle=True, \
         callbacks=[early_stopping, model_checkpoint])
         
model.load_weights(bst_model_path)

# model.load_weights('simple_lstm_glove_vectors_0.25_0.25_old.h5')
bst_val_score = min(hist.history['val_loss'])
print(bst_val_score)


print('Start making the submission before fine-tuning')

y_test = model.predict([test_data], batch_size=1024, verbose=1)

print(test_data[:50])
print(y_test[:50])
