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


def load_data():
    df = pd.read_excel('./1958至今發警報颱風.xlsx') 
    df['filter'] = pd.Series(np.ones(df.shape[0]))
    df = df[df['年份'] <=2015] # drop no img ty
    df['侵臺路徑分類'][df['侵臺路徑分類'] == '特殊'] = 0  # 特殊 == 0
    df['filter'][df['侵臺路徑分類'] == '---'] = 0 # 未分類不用 
    df = df[df['filter'] == 1]
    df = df.assign(new = lambda x: x.年份.astype(str) + x.英文名稱) # 合併年份名稱
    #df.assign(new = lambda x: str(df['年份']) + df['英文名稱'])
    df = df.sort_values('new')
    train = df[['new', '侵臺路徑分類']].values
    imgs = sorted(glob.glob('./歷史颱風路徑圖/*.png'))
    imgNames = list([name[30:-4] for name in imgs])

    # img can use
    use_img = []
    i=0
    for name in imgNames:
        if name in train:
            use_img.append(imgs[i])
        i+=1
    # label can use
    use_label = []
    for name in train:
        if name[0] in imgNames:
            use_label.append(name[1])
           
    # preprocessing
    x = np.array([cv2.imread(file) for file in use_img])
    X = np.zeros((len(use_label),1024,1024))
    for i in range(X.shape[0]):
        X[i] = rgb2gray(x[i])
    X = X[:,200:860:2,200:900:2] # crop and pooling 
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    y = np.array(use_label)
    
    X = X.astype('float32')
    X /= 255
    y = keras.utils.to_categorical(y, 10)
    
    # split 
    indices = np.arange(X.shape[0])
    nb_test_samples = int(0.2 * X.shape[0])
    random.seed(777)
    random.shuffle(indices)
    # X
    X = X[indices]
    X_train = X[2*nb_test_samples:]
    X_test = X[0:nb_test_samples]
    X_val = X[nb_test_samples:2*nb_test_samples]
    # y
    y = y[indices]
    y_train = y[2*nb_test_samples:]
    y_test = y[0:nb_test_samples]
    y_val = y[nb_test_samples:2*nb_test_samples]
    
    return X_train, X_test, X_val, y_train, y_test, y_val

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def CNN(img_rows, img_cols):
    model = Sequential()
    model.add(Convolution2D(20, (3,3), use_bias=True, padding='SAME', strides=1, activation='relu', input_shape=(img_rows,img_cols,1)))
    model.add(Convolution2D(20, (3,3), use_bias=True, padding='SAME', strides=1, activation='relu' ))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Convolution2D(40, (3,3), use_bias=True, padding='SAME', strides=1, activation='relu'))
    model.add(Convolution2D(40, (3,3), use_bias=True, padding='SAME', strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Convolution2D(40, (3,3), use_bias=True, padding='SAME', strides=1, activation='relu'))
    model.add(Convolution2D(40, (3,3), use_bias=True, padding='SAME', strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Convolution2D(40, (5,5), use_bias=True, padding='SAME', strides=1, activation='relu'))
    model.add(Convolution2D(40, (5,5), use_bias=True, padding='SAME', strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(5,5)))
    #model.add(Convolution2D(20, (6,6), use_bias=True, padding='SAME', strides=1, activation='selu', input_shape=(img_rows,img_cols,1)))
    #model.add(MaxPooling2D(pool_size=(6,6)))
    #model.add(Convolution2D(20, (6,6), use_bias=True, padding='SAME', strides=1, activation='selu', input_shape=(img_rows,img_cols,1)))
    #model.add(MaxPooling2D(pool_size=(6,6)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    return model

    
if __name__ == '__main__':
    
    X_train, X_test, X_val, y_train, y_test, y_val = load_data()
    #print(y_train, y_train.shape)
    #print(y_val, y_val.shape)

    #X_train = np.concatenate((X_train, X_train, X_train, X_train, X_train))
    #y_train = np.concatenate((y_train, y_train, y_train, y_train, y_train))

    # hyperparameter
    BATCH_SIZE = 12
    EPOCHS = 50
    img_rows, img_cols = X_train.shape[1], X_train.shape[2]
    print(X_train.shape)
    print(y_train.shape)
    #model
    model = CNN(img_rows, img_cols)
    model.compile(loss = keras.losses.categorical_crossentropy,
                  optimizer = keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])
    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val), 
              batch_size=BATCH_SIZE, 
              epochs=EPOCHS,
              shuffle=True)
    print(model.evaluate(X_test, y_test))
