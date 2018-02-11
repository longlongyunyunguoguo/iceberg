import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten,Dense,Dropout
from keras.optimizers import Adam

np.random.seed(777)
#dense_activate='sigmoid'
dense_activate='relu'
filter_number=32
band_number=4 #we use angle (75 X 75, same value) as the 4th band input
#mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
#reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

def initialize_CNN(dense_activate,kernel_loop,filter_number,learning_rate,kernel_size_len,drop_out,band_num):
    model = Sequential()
    for i in range(kernel_loop):
        model.add(Conv2D(filters=filter_number, kernel_size=(kernel_size_len,kernel_size_len),padding='same', input_shape=(75, 75, band_num), activation='relu'))
        #model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        model.add(Dropout(drop_out))
        filter_number*=2
    model.add(Flatten())
    model.add(Dense(128, activation=dense_activate))
    #model.add(BatchNormalization())
    model.add(Dropout(drop_out))
    model.add(Dense(2, activation='softmax'))
    #sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=learning_rate, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model

def CNN_model_generation():
    model=initialize_CNN(dense_activate='relu',kernel_loop=4,filter_number=32,learning_rate=0.0005,kernel_size_len=4,drop_out=0.15,band_num=4)
    return model