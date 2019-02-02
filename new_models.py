
def softmax(x, axis=1):
    """Softmax activation function."""
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')
        
        

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

inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size)(inp)
x = Bidirectional(CuDNNLSTM(32, return_sequences= True))(x)
x = BatchNormalization()(x)
x = Bidirectional(CuDNNLSTM(32, return_sequences= True))(x)
x = Dropout(0.25)(x)
context = one_step_attention(x)
context = Flatten()(context)
merged = BatchNormalization()(context)
merged = Dropout(0.25)(merged)
preds = Dense(1, activation= 'sigmoid')(merged)
model = Model(inputs = [inp], outputs= preds)
model.compile(loss='binary_crossentropy', optimizer= 'rmsprop', metrics= ['accuracy'])
print(model.summary())







inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size)(inp)
x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
## Train the model 
model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))