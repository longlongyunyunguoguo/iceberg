import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
###No filtering Band3 by 2-1
#SAR speckcle filter
def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2
    overall_variance = variance(img)
    img_weights = img_variance**2 / (img_variance**2 + overall_variance**2)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output
#image input normalization
def transform(df):
    images = []
    for i, row in df.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_2 - band_1
        #band_1 = lee_filter(band_1,10)
        #band_2 = lee_filter(band_2,10)
        #band_3 = lee_filter(band_3,10)
        band_1_norm = (band_1 - band_1.min()) / (band_1.max() - band_1.min())
        band_2_norm = (band_2 - band_2.min()) / (band_2.max() - band_2.min())
        band_3_norm = (band_3 - band_3.min()) / (band_3.max() - band_3.min())
        images.append(np.dstack((band_1_norm, band_2_norm,band_3_norm)))
    return np.array(images)
#augment by mirrorring
def augment(images):
    image_mirror_lr = []
    image_mirror_ud = []
    for i in range(0,images.shape[0]):
        band_1 = images[i,:,:,0]
        band_2 = images[i,:,:,1]
        band_3 = images[i,:,:,2]
        # mirror left-right
        band_1_mirror_lr = np.flip(band_1, 0)
        band_2_mirror_lr = np.flip(band_2, 0)
        band_3_mirror_lr = np.flip(band_3, 0)
        image_mirror_lr.append(np.dstack((band_1_mirror_lr, band_2_mirror_lr,band_3_mirror_lr)))
        # mirror up-down
        band_1_mirror_ud = np.flip(band_1, 1)
        band_2_mirror_ud = np.flip(band_2, 1)
        band_3_mirror_ud = np.flip(band_3, 1)
        image_mirror_ud.append(np.dstack((band_1_mirror_ud, band_2_mirror_ud,band_3_mirror_ud)))
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
    train0 = pd.read_json('./data/'+train_file)
    test = pd.read_json('./data/test.json')
    test_pred = pd.read_csv('./data/submission_1055.csv')
    test = test.join(test_pred.set_index('id'),on='id')
    test_real = test[test['inc_angle'].apply(lambda x: len(str(x).split('.')[1])<=4)]
    test_real_pseudo = test_real[test_real['is_iceberg'].apply(lambda x: x>0.99 or x<0.01)]
    test_real_pseudo['is_iceberg'] = np.around(test_real_pseudo['is_iceberg']).astype('int64')
    train = pd.concat([train0,test_real_pseudo],ignore_index=True)
    
    train = shuffle(train, random_state=777)
    all_X,all_y=data_processing(train)
    #split
    train_num = 1403
    train_X,train_y=add_more(all_X[:train_num,:,:,:],all_y[:train_num,:])
    x_val=all_X[train_num:,:,:,:]
    y_val=all_y[train_num:,:]
    return train_X,train_y,x_val,y_val