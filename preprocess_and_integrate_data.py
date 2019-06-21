from __future__ import print_function
import numpy as np
from keras.models import load_model
from keras import backend as BK
import cv2
import scipy.io as sio
from audio_proc import AudioProcessor

ap = AudioProcessor()
ap._init_()

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

# Function to get audio bottleneck from the autoencoder model
def get_activations(model, layer_in, layer_out, X_batch):
    get_activations = BK.function([model.layers[layer_in].input, BK.learning_phase()], [model.layers[layer_out].output])
    activations = get_activations([X_batch,0])
    return activations

print('Loading autoencoder model...')
config = BK.tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = BK.tf.Session(config=config)
model=load_model('auto_model/autoencoder.h5',custom_objects={'corr2_mse_loss': corr2_mse_loss})
model.load_weights('auto_model/autoencoder_weights.h5')

FRAME_WIDTH=int(128)
FRAME_HEIGHT=int(128)
PATH='Processed/'
window_size=6

# Functions to process video
def load_video(path):
    cap = cv2.VideoCapture(path)
    fc = 0
    ret = True
    frameHeight=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    frameWidth=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    buf = np.empty((frameHeight, frameWidth, frameCount), np.dtype('float32'))
    
    while (fc < frameCount and ret):
        ret, frame = cap.read()
        if not ret:
            break
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame=frame.astype('float32')
        frame = frame-np.amin(frame)
        frame = frame/np.amax(frame)
        buf[:,:,fc]=frame
        fc += 1
    cap.release()
    
    num_pads = window_size - fc%window_size
    padded_data=np.pad(buf,((0,0),(0,0),(0,num_pads)),'constant')
    return padded_data

#Instead of having full frames, we change it to the difference between the frames
def diff(buf_input):
    buf_input=np.pad(buf_input,((0,0),(0,0),(1,0)),'edge')
    buf_output=np.diff(buf_input,axis=2)
    return buf_output
    
#Divide full video data into slices of frames
def slice_video(video):
    num_slices = int(video.shape[3]/window_size)
    video_output =np.empty((num_slices,3,FRAME_HEIGHT,FRAME_WIDTH,window_size), np.dtype('float32'))
    start=0
    for i in range(0,num_slices):
        video_output[i,:,:,:,:]=video[:,:,:,start:start+window_size]
        start+=window_size
    return video_output

# Functions to process audio
def slice_audio_spec(audio_spec):
    num_slices = int(audio_spec.shape[1] / window_size)
    audio_output =np.empty((num_slices,audio_spec.shape[0],window_size), np.dtype('float32'))
    start=0
    for i in range(0,num_slices):
        audio_output[i,:,:]=audio_spec[:,start:start+window_size]
        start+=window_size
        if start>audio_spec.shape[1]-window_size:
            break
    return audio_output

def get_padded_spec(data, slices):
    num_pads = int(slices*window_size*4 - data.shape[1])
    padded_data=np.pad(data,((0,0),(0,num_pads)),'constant')
    tmp = np.reshape(padded_data.T, (int(padded_data.shape[1]/4), int(padded_data.shape[0]*4)))
    bottleneck=get_activations(model, 0, 12, tmp)[0]
    return bottleneck.T

def get_spec(path):
    tmp = ap.load_wav(path)
    return ap.melspectrogram(tmp)

# Function to load audio and video from a list of paths
def get_data(paths):
    views = ['1']
    i=0
    for row in paths:
        for j,view in enumerate(views):
            s,f = row.split(',')
            curr_path = 'video/'+s+'/'+view+'/s'+s+'_v'+view+'_u'+f+'.avi'
            tmp0=load_video(PATH+curr_path)
            vid_slices = tmp0.shape[2]/window_size
            
            diff_video=np.empty((3,tmp0.shape[0],tmp0.shape[1],tmp0.shape[2]))
            diff_video[0,:,:,:]=tmp0
            diff_video[1,:,:,:]=diff(tmp0)
            diff_video[2,:,:,:]=diff(diff_video[1,:,:,:])
            if j==0:
                data_curr = slice_video(diff_video)
            else:
                data_curr = np.concatenate((data_curr, slice_video(diff_video)),axis=1)
        if i==0:
            video_input = data_curr
        else:
            video_input = np.concatenate((video_input, data_curr), axis=0)
            
        curr_path = 'audio/'+s+'/s'+s+'_u'+f+'.wav'
        data = get_spec(PATH+curr_path)
        data=get_padded_spec(data, vid_slices)
        data=slice_audio_spec(data)
        if i==0:
            audio_input = data
        else:
            audio_input = np.concatenate((audio_input, data))
        
        i+=1
        if i%10==0:
            print(str(i)+'/'+str(len(paths)))
    audio_output=np.reshape(audio_input,(audio_input.shape[0],audio_input.shape[1]*audio_input.shape[2]))
    return video_input, audio_output

#Read file path from text file
#The text file contains path to testing speaker files in the last 600 lines, 
# rest are the paths to training speaker files
#Both training and testing paths are randomly shuffled
text_file = open(PATH+'list.txt', 'r')
lines = text_file.read().split('\n')
lines = lines[:-1]

#Split data
num_test= 600
num_train= len(lines) - num_test

train_lines = lines[:num_train]
test_lines = lines[num_train:]

#Define total no of files and paths to process per file
N=60
L=int(np.ceil(num_train/N))

#Save data
print('Saving training data')
for i in range(N):
    print('Saving data part'+str(i+1)+'...')
    start=i*L
    if i<N-1:
        end=(i+1)*L
    else:
        end=num_train
    print(str(start)+' to '+str(end))
    video_data, audio_data = get_data(train_lines[start:end])
    sio.savemat('final_data/preprocessed_data_final_part'+str(i+1)+'.mat', mdict={'video_input': video_data, 'audio_input' : audio_data})

print('Saving testing data')
N_V = int(np.ceil(num_test/L))

for i in range(N_V):
    print('Saving validation data part'+str(i+1)+'...')
    start = i*L
    if i<N_V-1:
        end = (i+1)*L
    else:
        end = num_test
    print(str(start)+' to '+str(end))
    video_data, audio_data = get_data(test_lines[start:end])
    sio.savemat('final_data/preprocessed_data_final_validation_part'+str(i+1)+'.mat', mdict={'video_input': video_data, 'audio_input' : audio_data})
        
