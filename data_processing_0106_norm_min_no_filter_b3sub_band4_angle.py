import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.ndimage.measurements import variance
###No filtering Band3 by 2-1
#SAR speckcle filter

#image input normalization
def transform(df):
    images = []
    for i, row in df.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_2 - band_1
        band_4 = np.ones((75,75))*row['inc_angle']
        #band_1 = lee_filter(band_1,10)
        #band_2 = lee_filter(band_2,10)
        #band_3 = lee_filter(band_3,10)
        band_1_norm = (band_1 - band_1.min()) / (band_1.max() - band_1.min())
        band_2_norm = (band_2 - band_2.min()) / (band_2.max() - band_2.min())
        band_3_norm = (band_3 - band_3.min()) / (band_3.max() - band_3.min())
        band_4_norm =  band_4 /90
        images.append(np.dstack((band_1_norm, band_2_norm,band_3_norm,band_4_norm)))
    return np.array(images)
#augment by mirrorring
def augment(images):
    image_mirror_lr = []
    image_mirror_ud = []
    for i in range(0,images.shape[0]):
        band_1 = images[i,:,:,0]
        band_2 = images[i,:,:,1]
        band_3 = images[i,:,:,2]
        band_4 = images[i,:,:,3]
        # mirror left-right
        band_1_mirror_lr = np.flip(band_1, 0)
        band_2_mirror_lr = np.flip(band_2, 0)
        band_3_mirror_lr = np.flip(band_3, 0)
        image_mirror_lr.append(np.dstack((band_1_mirror_lr, band_2_mirror_lr,band_3_mirror_lr,band_4)))
        # mirror up-down
        band_1_mirror_ud = np.flip(band_1, 1)
        band_2_mirror_ud = np.flip(band_2, 1)
        band_3_mirror_ud = np.flip(band_3, 1)
        image_mirror_ud.append(np.dstack((band_1_mirror_ud, band_2_mirror_ud,band_3_mirror_ud,band_4)))
    mirrorlr = np.array(image_mirror_lr)
    mirrorud = np.array(image_mirror_ud)
    images = np.concatenate((images, mirrorlr, mirrorud))
    return images
def data_processing(df):
    train_X = transform(df)
    train_y_iceberg = np.array(df['is_iceberg'])
    train_y=np.column_stack((train_y_iceberg,np.abs(train_y_iceberg-1)))
    return train_X,train_y
def add_more(train_X,train_y):
    train_X = augment(train_X)
    train_y = np.concatenate((train_y, train_y, train_y))
    return train_X,train_y
def train_data_generation(train_file):
    ##read in files
    train = pd.read_json('./data/'+train_file)
    train = shuffle(train, random_state=777)
    all_X,all_y=data_processing(train)
    #split
    train_num = 1403
    train_X,train_y=add_more(all_X[:train_num,:,:,:],all_y[:train_num,:])
    x_val=all_X[train_num:,:,:,:]
    y_val=all_y[train_num:,:]
    return train_X,train_y,x_val,y_val