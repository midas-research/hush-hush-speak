# hush-hush-speak



The Auxillary file containing the Reconstructed Audio for both English as well as Hindi Speech and the human annotations can be found at https://drive.google.com/open?id=1ZWS4L3SaZyb7SNwTaMpY96uJRYfFcVEG .

Alternate Link: https://drive.google.com/open?id=1bvY9_1OT4xzDELnRnPJpWUFRdzQj0lu3



Instructions

1. First  download OuluVS2 
2. Run https://github.com/midas-research/hush-hush-speak/blob/master/prepare_files.py
3. Create a audio_names.txt which contains all the audio files you want to train the audio autoencoder on and then run  https://github.com/midas-research/hush-hush-speak/blob/master/autoenc_train.py to train the audio autoencoder
4. For each view combination prepare a list.txt which contains the speaker id (sx where x is [1,53])  and the video number is (uy where y is (1,70) first 1-30 represent Oulu digits 31-60 represents phrases and 61-70 represents sentences) these will represent the whole dataset. Now in https://github.com/midas-research/hush-hush-speak/blob/master/preprocess_and_integrate_data.py set train_lines and test_lines there to the indexes where there is test and train data. Also set the  total no of files N=60 you want to split the training data into. Also set views = ['1’] to the required view combination. And run the preprocess_and_integrate_data.py to get the processed data. Note you have to run it again for each view combination. Also you would have to set views variable again. 
5.  Now run https://github.com/midas-research/hush-hush-speak/blob/master/train_video_enc.py to get the trained model and the predicted values for test set. 
6. Use https://github.com/midas-research/hush-hush-speak/blob/master/audio_proc.py  functions like melspectrogram and inv_melspectrogram  for further evaluation. 


Note: The code was validated for only the following package versions:
Tensorflow version 1.11, Keras 2.2.4,   Cuda compilation tools release 9.0 (V9.0.176)
