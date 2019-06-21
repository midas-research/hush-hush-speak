from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, noise
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as BK
from keras.regularizers import l1_l2
from keras.layers.normalization import BatchNormalization
from keras.optimizers import *
import scipy.io as sio
import random 
from audio_proc import AudioProcessor

#Initialize audio processing file
ap = AudioProcessor()
ap._init_()

#Hyperparameters
window_size=6*4
PATH = 'Processed/'

#Loss Function
def corr2_mse_loss(a,b):
    a = BK.tf.subtract(a, BK.tf.reduce_mean(a))
    b = BK.tf.subtract(b, BK.tf.reduce_mean(b))
    tmp1 = BK.tf.reduce_sum(BK.tf.multiply(a,a))
    tmp2 = BK.tf.reduce_sum(BK.tf.multiply(b,b))
    tmp3 = BK.tf.sqrt(BK.tf.multiply(tmp1,tmp2))
    tmp4 = BK.tf.reduce_sum(BK.tf.multiply(a,b))
    r = -BK.tf.divide(tmp4,tmp3)
    m=BK.tf.reduce_mean(BK.tf.square(BK.tf.subtract(a, b)))
    rm=BK.tf.add(r,m)
    return rm

#Functions for getting audio mel spectrogram
def get_padded_spec(data):
    t=data.shape[1]
    num_pads = window_size - t%window_size
    padded_data=np.pad(data,((0,0),(0,num_pads)),'constant')
    return padded_data

def get_spec(path):
    tmp = ap.load_wav(path)
    return ap.melspectrogram(tmp)

#Read file path from text file
text_file = open(PATH+'audio_names.txt', 'r')
lines = text_file.read().split('\n')
lines = lines[:-1]
index_shuf=list(range(len(lines)))
random.shuffle(index_shuf)

lines_shuf=[]
for i in index_shuf:
    lines_shuf.append(lines[i])

num_audios=len(lines)

#Load and preprocess audios
i=0
for row in lines_shuf:
    data = get_spec(PATH+row)
    data=get_padded_spec(data=data)
    tmp = np.reshape(data.T, (int(data.shape[1]/4), int(data.shape[0]*4)))
    if i==0:
        audio_input = tmp.astype(np.float32)
    else:
        audio_input = np.concatenate((audio_input, tmp.astype(np.float32)))
    i+=1
    if i%100==0:
        print(str(i)+'/'+str(num_audios))

print('zero check, values = ',np.count_nonzero(audio_input==0))
print('nan check, values = ',np.count_nonzero(np.isnan(audio_input)))

#Split train and test data
num_test = int(audio_input.shape[0]/5)
num_train = num_audios-num_test

audio_input_train=audio_input[:num_train,:]
audio_input_test=audio_input[num_train:,:]

print('Shape of all the data:'+str(audio_input.shape))
print('Shape of the train data to autoencoder:'+str(audio_input_train.shape))
print('Shape of the validation data:'+str(audio_input_test.shape))

#Define the model
config = BK.tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = BK.tf.Session(config=config)

adam=Adam(lr=.0001, clipnorm=1.)
model=Sequential()

model.add(Dense(1024, input_shape=(audio_input_train.shape[1],)))
model.add(BatchNormalization())
model.add(LeakyReLU())

model.add(Dense(256))
model.add(BatchNormalization())
model.add(LeakyReLU())

model.add(Dense(128))
model.add(BatchNormalization())
model.add(LeakyReLU())

model.add(Dense(64,kernel_regularizer=l1_l2(.001)))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
model.add(noise.GaussianNoise(.05))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(LeakyReLU())

model.add(Dense(256))
model.add(BatchNormalization())
model.add(LeakyReLU())

model.add(Dense(audio_input_train.shape[1]))
model.compile(loss=corr2_mse_loss, optimizer=adam)
model.summary()

#Model training
num_iter=200
loss_hist=np.empty((num_iter,2), dtype='float32')
for i in range(num_iter):
    print('Training autoencoder model, iteration: '+str(i)+'/'+str(num_iter))
    history = model.fit(audio_input_train, audio_input_train, batch_size=256, epochs=1, verbose=1, validation_data=(audio_input_test,audio_input_test))
    loss_hist[i,0]=history.history['loss'][0]
    loss_hist[i,1]=history.history['val_loss'][0]
    sio.savemat('auto_model/autoencoder_loss_history.mat', mdict={'history':loss_hist})

#Save the model and weights
model.save('auto_model/autoencoder.h5')
model.save_weights('auto_model/autoencoder_weights.h5')