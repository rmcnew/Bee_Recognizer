import numpy as np
import tflearn
from tflearn.data_utils import image_preloader
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

X, Y = image_preloader("/home/rmcnew/USU/intelligent_systems/Bee_Recognizer/BEE1/test/test_files.txt", image_shape=(32, 32), mode='file', categorical_labels=True, normalize=True, files_extension=['.png'])

X = np.reshape(X, (-1, 32, 32, 1))

convnet = input_data(shape=[None, 32, 32, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet,2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet,2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet,2,activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)

model.fit(X,Y, n_epoch=10, snapshot_step=500, show_metric=True, run_id='bee1_tflearn_test')

model.save('bee1_conv_test.model')
