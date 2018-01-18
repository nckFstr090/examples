import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Activation
from keras import losses
from keras import regularizers
from skimage import io
from skimage.transform import resize
from skimage.color import grey2rgb
from os.path import join
import os
import matplotlib.pyplot as plt
import cv2

POINTSNUM = 14
SPLITDATA = 25  # (1 / SPLITDATA) pictures in validation set
INPUTSHAPE = (224, 224)

def create_net():
    model = Sequential()
    model.add(Conv2D(32, (7, 7), padding="same", input_shape=(INPUTSHAPE[0], INPUTSHAPE[1], 3), kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(40, (5, 5), padding="same", kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(66, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(66, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(100, (2, 2), padding="same", kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(100, (2, 2), padding="same", kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())  # make output from previous layer 1-dimensional
    model.add(Dense(500, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation("relu"))
    
    model.add(Dropout(0.5))

    model.add(Dense(500, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation("relu"))
    
    model.add(Dropout(0.5))
    
    model.add(Dense(POINTSNUM * 2))
    return model

def resize_points(points, oldShape, newShape):
    for i in range(0, points.shape[0], 2):
        points[i] = points[i] * newShape[0] / oldShape[0]
        points[i + 1] = points[i + 1] * newShape[1] / oldShape[1]
    return points

def train_detector(trainPnts, trainImgDir, fast_train=False):
    model = create_net()
    model.compile(loss=losses.mean_squared_error, optimizer=keras.optimizers.Adadelta())

    files = np.array(list(trainPnts.keys()))
    batchSize = 64
    for ep in range(80):
        np.random.shuffle(files)
        
        pics = np.empty((files.shape[0], INPUTSHAPE[0], INPUTSHAPE[1], 3))
        for i in range(files.shape[0]):
            pic = io.imread(join(trainImgDir, files[i]))
            if len(pic.shape) != 3:
                pic = grey2rgb(pic)
            pics[i] = np.array(resize(pic, INPUTSHAPE, mode='reflect', preserve_range=True), dtype="uint8")
        shapes = [pic.shape[:2] for pic in pics]
        points = np.array([resize_points(trainPnts[files[i]], shapes[i], INPUTSHAPE) for i in range(len(files))])
        
        # training
        train = pics[: int(pics.shape[0] * ((SPLITDATA - 1) / SPLITDATA))]
        pointsTr = points[: int(points.shape[0] * ((SPLITDATA - 1) / SPLITDATA))]
        for s in range(train.shape[0] // batchSize + (train.shape[0] % batchSize != 0)):
            batch = train[s * batchSize : (s + 1) * batchSize]
            batchPoints = pointsTr[s * batchSize : (s + 1) * batchSize]
            model.train_on_batch(batch, batchPoints)
            if fast_train and s == 10:
                break

        # validation
        validation = pics[int(pics.shape[0] * ((SPLITDATA - 1) / SPLITDATA)) :]
        pointsVal = points[int(points.shape[0] * ((SPLITDATA - 1) / SPLITDATA)) :]
        for s in range(validation.shape[0] // batchSize + 1):
            batch = validation[s * batchSize : (s + 1) * batchSize]
            batchPoints = pointsVal[s * batchSize : (s + 1) * batchSize]
            model.test_on_batch(batch, batchPoints)

        if fast_train:
            break
    return model

def show(images, points):
    # show images with points
    x, y = 2, 2
    k = 0
    for i in range(images.shape[0]):
        plt.subplot(x, y, k + 1)
        k += 1
        #plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        p = points[i]
        p = p.reshape(p.shape[0] // 2, 2)
        plt.plot(p[:, 0], p[:, 1], "ro")
        if k == x * y:
            plt.show()
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            k = 0
    plt.show()

def detect(model, testImgDir):
    testImgDirList = os.listdir(testImgDir)
    picsNum = len(testImgDirList)
    test = np.empty((picsNum, INPUTSHAPE[0], INPUTSHAPE[1], 3))
    shapes = np.empty((picsNum, 2))
    for i in range(picsNum):
        pic = io.imread(join(testImgDir, testImgDirList[i]))
        if len(pic.shape) != 3:
            pic = grey2rgb(pic)
        shapes[i] = pic.shape[:2]
        pic = np.array(resize(pic, INPUTSHAPE, mode='reflect', preserve_range=True), dtype="uint8")
        test[i] = pic
    pred = model.predict(test, batch_size=1)
    output = {}
    for i in range(picsNum):
        output[testImgDirList[i]] = resize_points(pred[i], INPUTSHAPE, shapes[i])
    show(test, output)
    return output

