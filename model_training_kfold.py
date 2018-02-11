import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
import h5py
import data_processing_0101_norm_min_no_filter_b3sub_pseudo_1055 as data_pro
import cnn_model_1226 as cnn_model
from sklearn.model_selection import KFold
#trial id
id_num = 'no_angle_1055'
n = 5
#data processing
train = pd.read_json('drive/Deep_Learning/train.json')
all_X,all_y = data_pro.data_processing(train)
kf = KFold(n_splits=n, shuffle=True, random_state=2018)
ids=list(kf.split(all_X))
loc_i = np.arange(n)
test = pd.read_json('drive/Deep_Learning/test.json')
test_X = data_pro.transform(test)

def model_fitting(ids,I):
    #ids,I = inputs
    global test_X, all_X, all_y
    train_X = all_X[ids[0]]
    train_y = all_y[ids[0]]
    x_val = all_X[ids[1]]
    y_val = all_y[ids[1]]
    train_X,train_y = data_pro.add_more(train_X,train_y)
    #model fitting
    model = cnn_model.CNN_model_generation()
    epochs_number = 50
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('.mdl_'+str(id_num)+'_'+str(I)+'.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    history = History()
    model.fit(train_X,train_y, batch_size=32, epochs=epochs_number, verbose=1, validation_data=(x_val,y_val), shuffle=True, class_weight=None, initial_epoch=0, callbacks=[earlyStopping,mcp_save,history])
    df = pd.DataFrame.from_dict(history.history)
    df.to_csv('history_'+str(id_num)+'_'+str(I)+'.csv', sep='\t', index=True, float_format='%.4f')
    #test data prediction
    model.load_weights(filepath = '.mdl_'+str(id_num)+'_'+str(I)+'.hdf5')
    pred_test = model.predict(test_X)
    #save model parameters
    model.save('my_model_'+str(id_num)+'_'+str(I)+'.h5')
    return pred_test[:,0]

pred_outputs = []
for i in range(n):
    pred_outputs.append(model_fitting(ids[i],i))
final_pred = np.mean(np.array(pred_outputs),axis=0)
submission = pd.DataFrame({'id': test["id"], 'is_iceberg': final_pred})
submission.to_csv('submission_'+str(id_num)+'.csv', index=False)
