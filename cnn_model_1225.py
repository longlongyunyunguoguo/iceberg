import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten,Dense,Dropout
from keras.optimizers import Adam

np.random.seed(777)
#dense_activate='sigmoid'
dense_activate='relu'
filter_number=32
#mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
#reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

def initialize_CNN(dense_activate,kernel_loop,filter_number,learning_rate,kernel_size_len,drop_out):
    model = Sequential()
    for i in range(kernel_loop):
        model.add(Conv2D(filters=filter_number, kernel_size=(kernel_size_len,kernel_size_len),padding='same', input_shape=(75, 75, 3), activation='relu'))
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
    #5-fold cross validation
    #from sklearn.model_selection import KFold
    #from keras.callbacks import History 
    #CV=5
    #evaluation=np.zeros((CV,2))
    #kf = KFold(n_splits=CV,random_state=100)
    #kernel_loop_pool=[2,3,4]
    kernel_loop=4
    #lr_pool=[0.00001,0.00005,0.0001,0.0005,0.001]
    learning_rate=0.0005
    kernel_size_len=4
    #drop_out_pool=[0.05,0.1,0.15,0.2]
    drop_out=0.15
    print("kernel_loop: ",kernel_loop)
    print("learning_rate: ",learning_rate)
    print("kernel_size_len: ",kernel_size_len)
    print("drop_out: ",drop_out)
    model=initialize_CNN(dense_activate,kernel_loop,filter_number,learning_rate,kernel_size_len,drop_out)
    return model