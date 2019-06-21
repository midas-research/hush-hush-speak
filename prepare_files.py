import os
import cv2
import sys

## Input the range of speakers as arguments, e.g. python prepare_crop_files.py 1 50
if(len(sys.argv)<3):
	print('Insufficient arguments')
	quit()

start=int(sys.argv[1])
end=int(sys.argv[2])


path='../../lipsync/dataset'
res_path = 'Processed/'

if not os.path.exists(res_path):
	os.mkdir(res_path)
if not os.path.exists(res_path + 'video/'):
	os.mkdir(res_path + 'video/')
if not os.path.exists(res_path + 'audio/'):	
	os.mkdir(res_path + 'audio/')

for i in range(start,end+1):
	print('\nProcessing for speaker ',i,'\n')
	digit_source_path = path + '/cropped_mouth_mp4_digit/' + str(i) + '/'
	phrase_source_path = path + '/cropped_mouth_mp4_phrase/' + str(i) + '/'
	audio_source_path = path + '/cropped_audio_dat/' + str(i) + '/'

	if not os.path.exists(res_path + 'video/' + str(i) + '/'):
		os.mkdir(res_path + 'video/' + str(i) + '/')
	if not os.path.exists(res_path + 'audio/' + str(i) + '/'):
		os.mkdir(res_path + 'audio/' + str(i) + '/')

	for j in range(1,30+1):

		for view in range(1,5+1):
			if not os.path.exists(res_path + 'video/' + str(i) + '/' + str(view) + '/'):
				os.mkdir(res_path + 'video/' + str(i) + '/' + str(view) + '/')

			filename1 = digit_source_path + str(view) + '/' + 's' + str(i) + '_v' + str(view) + '_u' + str(j) + '.mp4'
			filename2 = phrase_source_path + str(view) + '/' + 's' + str(i) + '_v' + str(view) + '_u' + str(30+j) + '.mp4'
			cap1 = cv2.VideoCapture(filename1)
			cap2 = cv2.VideoCapture(filename2)
			out1 = cv2.VideoWriter()
			out2 = cv2.VideoWriter()
			fourcc=cv2.VideoWriter_fourcc('m','p','4','v')
			success1 = out1.open(res_path + 'video/' + str(i) + '/' + str(view) + '/' + 's' + str(i) + '_v' + str(view) + '_u' + str(j) + '.avi', fourcc, 29.97, (128,128),False)
			success2 = out2.open(res_path + 'video/' + str(i) + '/' + str(view) + '/' + 's' + str(i) + '_v' + str(view) + '_u' + str(30+j) + '.avi', fourcc, 29.97, (128,128),False)
			print('Digit Success: '+str(success1))
			print('Phrase Success: '+str(success2))

			while(cap1.isOpened()):
				ret, frame = cap1.read()
				if ret==False:
					break
				#Convert frame to grayscale
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				#Resize frame to (128,128)
				roi=cv2.resize(gray,(128,128))
				out1.write(roi)
			cap1.release()
			out1.release()

			while(cap2.isOpened()):
				ret, frame = cap2.read()
				if ret==False:
					break
				#Convert frame to grayscale
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				#Resize frame to (128,128)
				roi=cv2.resize(gray,(128,128))
				out2.write(roi)
			cap2.release()
			out2.release()

		#Process audio
		digit_audio_filename = audio_source_path + 's' + str(i) + '_u' + str(j) + '.wav'
		phrase_audio_filename = audio_source_path + 's' + str(i) + '_u' + str(30+j) + '.wav'
		os.system('ffmpeg -i '+ digit_audio_filename +' -ac 1 -ar 22050 ' + res_path+'audio/'+str(i)+'/s'+str(i) +'_u'+str(j)+'.wav' )
		os.system('ffmpeg -i '+ phrase_audio_filename +' -ac 1 -ar 22050 ' + res_path+'audio/'+str(i)+'/s'+str(i) +'_u'+str(30+j)+'.wav' )


