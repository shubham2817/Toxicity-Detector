

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

from nmt_utils import *

import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import numpy as np
from nltk.corpus import stopwords
import re


from gensim.models import KeyedVectors



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

print(len(embeddings_index))
print train_df.shape, test_df.shape
print(train_df.head())
print(test_df.head())





# # ====================================================================================================================
# # ====================================================================================================================
# # Preprocessing when using embeddings

# def build_vocab(sentences, verbose =  True):
#     vocab = {}
#     for sentence in tqdm(sentences, disable = (not verbose)):
#         for word in sentence:
#             try:
#                 vocab[word] += 1
#             except KeyError:
#                 vocab[word] = 1
#     return vocab


# sentences = train_df["comment_text"].progress_apply(lambda x: x.split()).values
# vocab = build_vocab(sentences)
# print(len(vocab))
# print({k: vocab[k] for k in list(vocab)[:5]})




# import operator 

# def check_coverage(vocab,embeddings_index):
#     a = {}
#     oov = {}
#     k = 0
#     i = 0
#     for word in tqdm(vocab):
#         # print(word)
#         try:
#             # print(word, len(embeddings_index[word]))
#             a[word] = embeddings_index[word]
#             k += vocab[word]
#             # print(word, len(embeddings_index[word]), k)
#         except:

#             oov[word] = vocab[word]
#             i += vocab[word]
#             pass
#     print(len(a)*100 / len(vocab))
#     print(k*100 / (k + i))
#     print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
#     print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
#     sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]
#     print(len(sorted_x))
#     return sorted_x

# oov = check_coverage(vocab,embeddings_index)
# print(oov[:10])





# print('?' in embeddings_index)
# print('&' in embeddings_index)


# def clean_text(x):
#     x = str(x)
#     for punct in "/-'":
#         x = x.replace(punct, ' ')
#     for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '""':
#         x = x.replace(punct, '')
#     return x

# train_df["comment_text"] = train_df["comment_text"].progress_apply(lambda x: clean_text(x)).values
# sentences = train_df["comment_text"].apply(lambda x: x.split())
# vocab = build_vocab(sentences)
# # print(len(vocab))
# # print({k: vocab[k] for k in list(vocab)[:5]})


# oov = check_coverage(vocab,embeddings_index)
# print(oov[:10])





# def clean_numbers(x):

#     x = re.sub('[0-9]{5,}', '#####', x)
#     x = re.sub('[0-9]{4}', '####', x)
#     x = re.sub('[0-9]{3}', '###', x)
#     x = re.sub('[0-9]{2}', '##', x)
#     return x

# train_df["comment_text"] = train_df["comment_text"].progress_apply(lambda x: clean_numbers(x)).values
# sentences = train_df["comment_text"].progress_apply(lambda x: x.split())
# vocab = build_vocab(sentences)
# # print(len(vocab))
# # print({k: vocab[k] for k in list(vocab)[:5]})


# oov = check_coverage(vocab,embeddings_index)
# print(oov[:10])    









# def _get_mispell(mispell_dict):
#     mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
#     return mispell_dict, mispell_re


# mispell_dict = {'colour':'color',
#                 'centre':'center',
#                 'didnt':'did not',
#                 'doesnt':'does not',
#                 'isnt':'is not',
#                 'shouldnt':'should not',
#                 'favourite':'favorite',
#                 'travelling':'traveling',
#                 'counselling':'counseling',
#                 'theatre':'theater',
#                 'cancelled':'canceled',
#                 'labour':'labor',
#                 'organisation':'organization',
#                 'wwii':'world war 2',
#                 'citicise':'criticize',
#                 'instagram': 'social medium',
#                 'whatsapp': 'social medium',
#                 'snapchat': 'social medium'

#                 }
# mispellings, mispellings_re = _get_mispell(mispell_dict)

# def replace_typical_misspell(text):
#     def replace(match):
#         return mispellings[match.group(0)]

#     return mispellings_re.sub(replace, text)


# train_df["comment_text"] = train_df["comment_text"].progress_apply(lambda x: replace_typical_misspell(x)).values
# sentences = train_df["comment_text"].progress_apply(lambda x: x.split())
# vocab = build_vocab(sentences)
# # print(len(vocab))
# # print({k: vocab[k] for k in list(vocab)[:5]})
# to_remove = ['a','to','of','and']
# sentences = [[word for word in sentence if not word in to_remove] for sentence in tqdm(sentences)]
# vocab = build_vocab(sentences)


# oov = check_coverage(vocab,embeddings_index)
# print(oov[:20])        

# # ====================================================================================================================
# # ====================================================================================================================






# print(sentences[:5])


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







# ================================================================================


def build_vocab(sentences, verbose =  True):
    vocab = {}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


df = pd.Series( (v for v in comments) )
print(len(df),"======")
print(df[:10])

sentences = df.progress_apply(lambda x: x.split()).values
vocab = build_vocab(sentences)
print(len(vocab))
print({k: vocab[k] for k in list(vocab)[:5]})




import operator 

def check_coverage(vocab,embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    c_try = 0
    c_catch = 0
    total = 0
    for word in tqdm(vocab):
        # print(word)
        total +=1
        try:
            # print(word, len(embeddings_index[word]))
            a[word] = embeddings_index[word]
            k += vocab[word]
            c_try +=1
            # print(word, len(embeddings_index[word]), k)
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            c_catch +=1
            pass
        print(c_try,c_catch,total)            
    print(len(a)*100 / len(vocab))
    print(k*100 / (k + i))
    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]
    print(len(sorted_x))
    return sorted_x

oov = check_coverage(vocab,embeddings_index)
print(oov[:10])


# ================================================================================




'''


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
activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
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
x = cudnn_lstm_layer(embedded_sequences)
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








plot_model(model, to_file='trial/sentiment_model_plot_diff_without_attention_bi_lstm.png', show_shapes=True, show_layer_names=True)


STAMP = 'trial/simple_lstm_glove_vectors_diff_without_attention_bi_lstm_%.2f_%.2f'%(rate_drop_lstm,rate_drop_dense)
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

#######################################
## make the submission
########################################
print('Start making the submission before fine-tuning')

y_test = model.predict([test_data], batch_size=1024, verbose=1)

print(test_data[:50])
print(y_test[:50])

'''