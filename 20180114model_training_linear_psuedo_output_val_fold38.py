import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
import h5py
import ProcessingMethods.data_processing_0106_norm_min_no_filter_b3sub_band4_angle as data_pro
import cnn_model_0106_angle_4thband as cnn_model
from sklearn.model_selection import KFold
import LinearPred_Angle as linear_predict
#trial id
id_num = 3209
n = 10
#data processing+pseudo coding
train0 = pd.read_json('train.json')
train0['inc_angle'] = pd.to_numeric(train0['inc_angle'],errors='coerce')
train_known=train0[train0.inc_angle.notnull()]
train_unknown=train0[train0.inc_angle.isnull()]

test = pd.read_json('test.json')
train_unknown.inc_angle = linear_predict.LinearPred(train_known,train_unknown,test)
train0 = pd.concat([train_known,train_unknown])
train0 = train0.sort_index()

test_pred = pd.read_csv('submission_3016')
test = test.join(test_pred.set_index('id'),on='id')
test_real = test[test['inc_angle'].apply(lambda x: len(str(x).split('.')[1])<=4)]
test_real_pseudo = test_real[test_real['is_iceberg'].apply(lambda x: x>0.99 or x<0.01)]
test_real_pseudo['is_iceberg'] = np.around(test_real_pseudo['is_iceberg']).astype('int64')
#train = pd.concat([train0,test_real_pseudo],ignore_index=True)

all_X,all_y = data_pro.data_processing(train0)
new_X,new_y = data_pro.data_processing(test_real_pseudo)
kf = KFold(n_splits=n, shuffle=True, random_state=2018)
ids=list(kf.split(all_X))
ids_new = list(kf.split(new_X))
loc_i = np.arange(n)
test = pd.read_json('test.json')
test_X = data_pro.transform(test)
pred_val=np.zeros([all_X.shape[0],])
def model_fitting(ids,ids_new,I,pred_val):
    #ids,I = inputs
    global test_X, all_X, all_y, new_X, new_y
    train_X = np.concatenate((all_X[ids[0]],new_X[ids_new[0]]),axis=0)
    train_y = np.concatenate((all_y[ids[0]],new_y[ids_new[0]]),axis=0)
    x_val = all_X[ids[1]]
    y_val = all_y[ids[1]]
    train_X,train_y = data_pro.add_more(train_X,train_y)
    #model fitting
    model = cnn_model.initialize_CNN(dense_activate='relu',kernel_loop=5,filter_number=32,learning_rate=0.001,kernel_size_len=3,drop_out=0.2,band_num=4)
    epochs_number = 100
    earlyStopping = EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('.mdl_'+str(id_num)+'_'+str(I)+'.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    history = History()
    model.fit(train_X,train_y, batch_size=64, epochs=epochs_number, verbose=1, validation_data=(x_val,y_val), shuffle=True, class_weight=None, initial_epoch=0, callbacks=[earlyStopping,mcp_save,history])
    df = pd.DataFrame.from_dict(history.history)
    df.to_csv('history_'+str(id_num)+'_'+str(I)+'.csv', sep='\t', index=True, float_format='%.4f')
    #test data prediction
    model.load_weights(filepath = '.mdl_'+str(id_num)+'_'+str(I)+'.hdf5')
    pred_test = model.predict(test_X)
    tmp = model.predict(x_val)
    pred_val[ids[1]]=tmp[:,0]##just keep first column
    submission = pd.DataFrame({'id': test["id"], 'is_iceberg': pred_test[:,0]})##not final_pred
    submission.to_csv('submission_'+str(id_num)+'_'+str(I)+'.csv', index=False)##each should have a different name
    #save model parameters
    #model.save('my_model_'+str(id_num)+'_'+str(I)+'.h5')
    return pred_test[:,0],pred_val

pred_outputs = []
pred_single=[]
for i in np.arange(38,39,1):
    pred_single,pred_val=model_fitting(ids[i],ids_new[i],i,pred_val)
    pred_outputs.append(pred_single)
#final_pred = np.mean(np.array(pred_outputs),axis=0)
#validation_pred=pd.DataFrame({'id': train0["id"], 'is_iceberg': pred_val})
#validation_pred.to_csv('validation_'+str(id_num)+'.csv', index=False)
#submission = pd.DataFrame({'id': test["id"], 'is_iceberg': final_pred})
#submission.to_csv('submission_'+str(id_num)+'.csv', index=False)