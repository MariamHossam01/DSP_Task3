import os
from flask import Flask, render_template, request
import numpy as np
import pickle
import librosa
import pyaudio
import wave
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture 
from sklearn import preprocessing
from scipy.io.wavfile import read
from Draw import*
app = Flask(__name__,template_folder="templates")


def record():
		FORMAT = pyaudio.paInt16
		CHANNELS = 1
		RATE = 44100
		CHUNK = 512
		RECORD_SECONDS = 2.5
		audio = pyaudio.PyAudio()
		stream = audio.open(format=FORMAT, channels=CHANNELS,
						rate=RATE, input=True,input_device_index = 1,
						frames_per_buffer=CHUNK)
		Recordframes = []
		for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
			data = stream.read(CHUNK)
			Recordframes.append(data)
		stream.stop_stream()
		stream.close()
		audio.terminate()
		WAVE_OUTPUT_FILENAME=os.path.join("audio","audio.wav")
		waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
		waveFile.setnchannels(CHANNELS)
		waveFile.setsampwidth(audio.get_sample_size(FORMAT))
		waveFile.setframerate(RATE)
		waveFile.writeframes(b''.join(Recordframes))
		waveFile.close()


def load_speech_model(x):
   model = pickle.load(open("trainedModel.sav",'rb')) 
   y=model.predict(np.array(x).reshape(1,-1))[0]
   return y

def extractWavFeatures():
   list_of_features=[]
   y, sr = librosa.load('audio/audio.wav', mono=True, duration=2.5)
   # remove leading and trailing silence
   y, index = librosa.effects.trim(y)

   chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
   rmse = librosa.feature.rms(y=y)
   spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
   spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
   rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
   zcr = librosa.feature.zero_crossing_rate(y)
   mfcc = librosa.feature.mfcc(y=y, sr=sr)

   list_of_features.append(np.mean(chroma_stft))
   list_of_features.append(np.mean(rmse))
   list_of_features.append(np.mean(spec_cent))
   list_of_features.append(np.mean(spec_bw))
   list_of_features.append(np.mean(rolloff))
   list_of_features.append(np.mean(zcr))

   for e in mfcc:
         list_of_features.append(np.mean(e))
   
   return(list_of_features)

def calculate_delta(array):

   rows,cols = array.shape
   print(rows)
   print(cols)
   deltas = np.zeros((rows,20))
   N = 2
   for i in range(rows):
      index = []
      j = 1
      while j <= N:
         if i-j < 0:
            first =0
         else:
            first = i-j
         if i+j > rows-1:
            second = rows-1
         else:
            second = i+j 
         index.append((second,first))
         j+=1
      deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
   return deltas


def extract_features(audio,rate):
   
   mfcc_feature = mfcc.mfcc(audio,rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True)    
   mfcc_feature = preprocessing.scale(mfcc_feature)
   # print(mfcc_feature)
   delta = calculate_delta(mfcc_feature)
   combined = np.hstack((mfcc_feature,delta)) 
   return combined

def speaker_model():

   sr,audio = read('audio/audio.wav')
   vector   = extract_features(audio,sr)

   gmm_files    = ['gmm_models/dina.gmm','gmm_models/kareman.gmm','gmm_models/mariam.gmm','gmm_models/nada.gmm']
   models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
   log_likelihood = np.zeros(len(models)) 

   for i in range(len(models)):
      gmm    = models[i]  #checking with each model one by one
      scores = np.array(gmm.score(vector))
      log_likelihood[i] = scores.sum()
   print(log_likelihood)
   winner = np.argmax(log_likelihood)

   if max(log_likelihood)<-24:
      return 'others'
   
   speakers=['Dina','Kareman','Mariam','Nada']
   return speakers[winner]


# def load_sound_model(x):
#    model = pickle.load(open("trained_speaker_model.sav",'rb')) 
#    y=model.predict(np.array(x).reshape(1,-1))[0]
#    return y


@app.route('/', methods=['GET', 'POST'])
def speechRecognation():
   
   record()

   speech_features=[]
   speech_features.append(extractWavFeatures())
   words=load_speech_model(speech_features)
   if words==0:
      word='open the door'
   else:
      word='others'

   speaker=speaker_model()

   print(f'speaker{speaker},words{words}')



   spectrum= visualize("audio/audio.wav")
   spectrum='static/assets/images/result'+str(variables.counter)+'.jpg'
   variables.counter+=1
   spec_fig=spectral_features("audio/audio.wav")
   spec_fig='static/assets/images/spec'+str(variables.counter)+'.jpg'
   mfcc_fig=Mfcc("audio/audio.wav")
   mfcc_fig='static/assets/images/mfcc'+str(variables.counter)+'.jpg'

   
   if speaker=='others':
      return render_template('index.html',words='Unknown Person',spectrum=spectrum,spec_fig=spec_fig,mfcc_fig=mfcc_fig)
   elif word=='others':
      return render_template('index.html',words=f'Hello {speaker}, you entered the wrong password.',spectrum=spectrum,spec_fig=spec_fig,mfcc_fig=mfcc_fig)
   else:
      return render_template('index.html',words=f'Welcome {speaker}!',spectrum=spectrum,spec_fig=spec_fig,mfcc_fig=mfcc_fig)

   

if __name__ == '__main__':
   app.run(debug=True)