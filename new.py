import glob, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import Counter

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, Dense, Dropout, Convolution2D, MaxPooling2D
from keras.callbacks import *
from keras.layers.normalization import BatchNormalization
from keras import optimizers
import keras 

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def load_data(flag):
    #  part1  read .xlsx data
    df = pd.read_excel('./1958至今發警報颱風.xlsx') 
    df['filter'] = pd.Series(np.ones(df.shape[0]))
    #df = df[df['年份'] <=2015]                                    # drop no img ty
    df['侵臺路徑分類'][df['侵臺路徑分類'] == '特殊'] = 0               # 特殊 == 0
    df['filter'][df['侵臺路徑分類'] == '---'] = 0                   # 未分類不用 
    df = df[df['filter'] == 1]
    df = df.assign(new = lambda x: x.年份.astype(str) + x.英文名稱) # 合併年份名稱 方便之後的圖片篩選 （因為圖片的檔名）
    df = df.sort_values('new')                                    # 依照新的標記排序
    train = df[['new', '侵臺路徑分類']].values                      # 把資料變成 numpy array
    
    # part2 read img data
    imgs = sorted(glob.glob('./歷史颱風路徑圖/*.png'))
    imgNames = list([name[19:-4] for name in imgs])               # imgNames 用以確認沒有取錯資料
    imgs2 = sorted(glob.glob('./歷史颱風路徑圖_標記起始點/*.png'))    # 有標記的檔案也做一樣的事情 
    imgNames2 = list([name[25:-4] for name in imgs2])

    # img can use 把可以用的img取出
    use_img = []
    i=0
    for name in imgNames:
        if name in train:
            use_img.append(imgs[i])
        i+=1

    use_img2 = []
    i=0
    for name in imgNames2:
        if name in train:
            use_img2.append(imgs2[i])
        i+=1    
    # label can use  把可以用的label取出
    use_label = []
    for name in train:
        if name[0] in imgNames:
            use_label.append(name[1])
    
    #use_label = use_label*2
    #use_feature = use_img + use_img2         

    #return use_img, use_img2, use_label     # 原本想寫成兩個function的 但是為了節省記憶體 我把他合成一個      

    #def Preprocessing():           
    # preprocessing  前處理
    x = np.array([cv2.imread(file) for file in use_img])
    X = np.zeros((len(use_label),1024,1024))
    for i in range(X.shape[0]):
        X[i] = rgb2gray(x[i])
    X = X[:,200:860:2,200:900:2] # crop and pooling 
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    y = np.array(use_label)
    
	# if flag == 'westgo'
    west = [1,2,3,4,5]
    if flag == 'westgo':
        for i in range(len(y)):
            if y[i] in west:
                y[i] = 0
            else:
                y[i] = 1

    X = X.astype('float32')
    X /= 255
    y = keras.utils.to_categorical(y, 10)
    
    # split 
    indices = np.arange(X.shape[0])
    nb_test_samples = int(0.2 * X.shape[0])
    random.seed(66)
    random.shuffle(indices)
    # X
    X = X[indices]
    X_train = X[nb_test_samples:]
    X_test = X[0:nb_test_samples]
    # y
    y = y[indices]
    y_train = y[nb_test_samples:]
    y_test = y[0:nb_test_samples]
    return X_train, X_test, y_train, y_test
    

def CNN(img_rows, img_cols):
    model = Sequential()
    model.add(Convolution2D(20, (3,3), use_bias=True, padding='SAME', strides=1, activation='selu', input_shape=(img_rows,img_cols,1)))
    model.add(Convolution2D(20, (3,3), use_bias=True, padding='SAME', strides=1, activation='selu' ))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Convolution2D(40, (3,3), use_bias=True, padding='SAME', strides=1, activation='selu'))
    model.add(Convolution2D(40, (3,3), use_bias=True, padding='SAME', strides=1, activation='selu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Convolution2D(40, (3,3), use_bias=True, padding='SAME', strides=1, activation='selu'))
    model.add(Convolution2D(40, (3,3), use_bias=True, padding='SAME', strides=1, activation='selu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Convolution2D(40, (5,5), use_bias=True, padding='SAME', strides=1, activation='selu'))
    model.add(Convolution2D(40, (5,5), use_bias=True, padding='SAME', strides=1, activation='selu'))
    model.add(MaxPooling2D(pool_size=(5,5)))

    model.add(Flatten())
    model.add(Dense(128, activation='selu'))
    model.add(Dense(128, activation='selu'))
    model.add(Dense(10, activation='softmax'))
    
    return model

if __name__ == '__main__':
    flag = 'westgo'
    X_train, X_test, y_train, y_test= load_data(flag)

    # hyperparameter
    BATCH_SIZE = 12
    EPOCHS = 50
    learning_rate = 0.0001
    
    img_rows, img_cols = X_train.shape[1], X_train.shape[2]
    print(X_train.shape)
    print(y_train.shape)
    #model
    model = CNN(img_rows, img_cols)
    model.compile(loss = keras.losses.categorical_crossentropy,
                  optimizer = keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])

    model.fit(X_train, y_train, 
              validation_data=(X_test, y_test), 
              batch_size=BATCH_SIZE, 
              epochs=EPOCHS,
              shuffle=True)
    print(model.evaluate(X_test, y_test))
