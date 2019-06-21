from __future__ import print_function
import numpy as np
import gc
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution3D, MaxPooling3D, LSTM
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as BK
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.optimizers import *
import scipy.io as sio

# Define loss function
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

# Hyper-parameters
window_size=6
FRAME_HEIGHT = 128
FRAME_WIDTH = 128
bottleneck_size = 64
tot_slices_train = 39762
tot_slices_val = 9229
views = ['1']
NUM_TRAIN=60
NUM_TEST = 15

video_input_shape = [tot_slices_train, 3*len(views), FRAME_HEIGHT, FRAME_WIDTH, window_size]
audio_input_shape = [tot_slices_train, bottleneck_size, window_size]

# Function to augment data
def data_augmentation(video):
    augmentation_type=[1,2,3]
    video=np.transpose(video,axes=[2,3,1,4,0])
    for i in range(video.shape[4]):
        a_type=np.random.choice(augmentation_type)
        if a_type==1: #Flip
            video[:,:,:,:,i]=np.fliplr(video[:,:,:,:,i])
            continue
        if a_type==2: #Noise
            video[:,:,:,:,i]+=np.random.normal(0,.01,(video[:,:,:,:,i].shape))
            continue
        if a_type==3:
            continue #Original
    video=np.transpose(video,axes=[4,2,0,1,3])
    return video

# Train data generator
trn_batch=64
steps_per_epoch=int(np.ceil(video_input_shape[0]/trn_batch))

def generate_train_data():
    while(1):
        for j in range(1,NUM_TRAIN+1):
            mat_tmp=sio.loadmat('final_data/preprocessed_data_final_part'+str(j)+'.mat')
            video_input = mat_tmp['video_input']
            audio_output = mat_tmp['audio_input']
            del mat_tmp
            gc.collect()
            audio_output=np.reshape(audio_output,(audio_output.shape[0],audio_input_shape[1]*audio_input_shape[2]))
            k=0
            while(1):
                if (k+int(trn_batch))>video_input.shape[0]:
                    augmented_vid=data_augmentation(video_input[k:,:,:,:,:])
                    yield (augmented_vid,audio_output[k:,:])
                    break
                else:
                    augmented_vid=data_augmentation(video_input[k:k+int(trn_batch),:,:,:,:])
                    yield (augmented_vid,audio_output[k:k+int(trn_batch),:])
                k+=int(trn_batch)

# Load validation+testing data
for i in range(1,NUM_TEST+1):
    print('Reading testing data, part ',str(i))
    mat=sio.loadmat('final_data/preprocessed_data_final_validation_part'+str(i)+'.mat')
    if i==1:
        video_input_test=mat['video_input']
        audio_input_test=mat['audio_input']
    else:
        video_input_test = np.concatenate((video_input_test,mat['video_input']),axis = 0)
        audio_input_test = np.concatenate((audio_input_test,mat['audio_input']),axis = 0)

#To get 85:10:5, we take 2/3 for validation and rest for testing
nb_v=video_input_test.shape[0]
nb_2half=int(np.floor((2*nb_v)/3))
video_input_validation=video_input_test[:nb_2half,:]
audio_input_validation=audio_input_test[:nb_2half,:]
video_input_test=video_input_test[nb_2half:,:]
audio_input_test=audio_input_test[nb_2half:,:]
mat=None

# Validation data generator
val_batch = 64
val_steps = np.ceil(video_input_validation.shape[0] / val_batch)

def generate_valid_data():
    vk = 0
    while(1):
        if vk + val_batch > video_input_validation.shape[0]:
            vid_data = video_input_validation[vk:,:,:,:,:]
            aud_data = audio_input_validation[vk:,:]
            vk = 0
        else:
            vid_data = video_input_validation[vk : vk+val_batch,:,:,:,:]
            aud_data = audio_input_validation[vk : vk+val_batch,:]
            vk += val_batch
        yield vid_data, aud_data


# Define Video encoder model
config = BK.tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = BK.tf.Session(config=config)

learn_rate = (.0001)
adam=Adam(lr=learn_rate)
reg=.0005
model=Sequential()

# Conv layer 1
model.add(Convolution3D(filters = 32, kernel_size=[3, 3, 3], input_shape=video_input_shape[1:],
                  data_format='channels_first', kernel_initializer="he_normal",padding='same', kernel_regularizer=l2(reg)))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPooling3D(pool_size=(2, 2, 1), data_format='channels_first'))

# Conv layer 2
model.add(Convolution3D(filters = 32, kernel_size=[3, 3, 3],data_format='channels_first', kernel_initializer="he_normal",padding='same', kernel_regularizer=l2(reg)))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPooling3D(pool_size=(2, 2, 1), data_format='channels_first'))

# Conv layer 3
model.add(Convolution3D(filters = 32, kernel_size=[3, 3, 3],data_format='channels_first', kernel_initializer="he_normal",padding='same', kernel_regularizer=l2(reg)))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPooling3D(pool_size=(2, 2, 1), data_format='channels_first'))
model.add(Dropout(.25))

# Conv layer 4
model.add(Convolution3D(filters = 64, kernel_size=[3, 3, 3],data_format='channels_first', kernel_initializer="he_normal",padding='same', kernel_regularizer=l2(reg)))
model.add(BatchNormalization())
model.add(LeakyReLU())

# Conv layer 5
model.add(Convolution3D(filters = 64, kernel_size=[3, 3, 3],data_format='channels_first', kernel_initializer="he_normal",padding='same', kernel_regularizer=l2(reg)))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPooling3D(pool_size=(2, 2, 1), data_format='channels_first'))
model.add(Dropout(.25))

# Conv layer 6
model.add(Convolution3D(filters = 128, kernel_size=[3, 3, 3],data_format='channels_first', kernel_initializer="he_normal",padding='same', kernel_regularizer=l2(reg)))
model.add(BatchNormalization())
model.add(LeakyReLU())

# Conv layer 7
model.add(Convolution3D(filters = 128, kernel_size=[3, 3, 3],data_format='channels_first', kernel_initializer="he_normal",padding='same', kernel_regularizer=l2(reg)))
model.add(BatchNormalization())
model.add(ELU(alpha=1.0))
model.add(MaxPooling3D(pool_size=(2, 2, 1), data_format='channels_first'))
model.add(Dropout(.25))

# Reshape for the LSTM layer
shape=model.get_output_shape_at(0)
model.add(Reshape((shape[-1],shape[1]*shape[2]*shape[3])))

# LSTM layer
model.add(LSTM(512, return_sequences=True, kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(ELU(alpha=1.0))
model.add(Dropout(.25))

model.add(Flatten())

# Dense layer 1
model.add(Dense(2048,kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(ELU(alpha=1.0))
model.add(Dropout(.4))

# Output layer
model.add(Dense(audio_input_shape[1]*audio_input_shape[2],kernel_initializer="he_normal",use_bias=True))
model.add(Activation('sigmoid'))

model.compile(loss=corr2_mse_loss,optimizer=adam)
model.summary()

# Load the best model when training in parts
#print('Loading the best model so far...')
#model.load_weights('models/Best_weights_LipReading.h5')

# Model training
num_iter=120

# Save original and predicted testing files for later testing
sio.savemat('models/test_orig_encoded.mat', mdict={'encode': np.reshape(audio_input_test,(audio_input_test.shape[0],audio_input_shape[1],audio_input_shape[2]))})

predict_final = np.empty((num_iter,audio_input_test.shape[0],audio_input_shape[1],audio_input_shape[2]), dtype='float32')

filepath="models/Best_weights_LipReading.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
loss_history=np.empty((num_iter,2), dtype='float32')

for i in range(num_iter):
    print('Training Video Encoder Model, iteration: '+str(i)+'/'+str(num_iter))
    history = model.fit_generator(generator=generate_train_data(),steps_per_epoch=steps_per_epoch, callbacks=callbacks_list, validation_data=generate_valid_data(), validation_steps=val_steps, epochs=1, verbose=1, max_q_size=10)
    predict = model.predict(video_input_test, batch_size=64)
    predict = np.reshape(predict,(predict.shape[0],audio_input_shape[1],audio_input_shape[2]))
    predict_final[i,:,:,:] = predict
    loss_history[i,0]=history.history['loss'][0]
    loss_history[i,1]=history.history['val_loss'][0]
    if i>3:
        if loss_history[i-4,1]<loss_history[i,1] and loss_history[i-4,1]<loss_history[i-1,1] and loss_history[i-4,1]<loss_history[i-2,1] and loss_history[i-4,1]<loss_history[i-3,1]:
            print("Loss didn't improve after 4 epochs, Dividing LR by 3")
            BK.set_value(model.optimizer.lr, .33*BK.get_value(model.optimizer.lr))
            learn_rate*= .33

    sio.savemat('models/history.mat', mdict={'encode': predict_final, 'history':loss_history})
    if i%5==0:
        model.save('models/model_LipReading.h5')
        model.save_weights('models/LipReading_mid_weights.h5')

print('Final lr = ',learn_rate)