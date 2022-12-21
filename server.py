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
   audio, sample_rate = librosa.load('audio/audio.wav', mono=True, duration=2.5)
   # remove leading and trailing silence
   audio, index = librosa.effects.trim(audio)

   chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
   rmse = librosa.feature.rms(y=audio)
   spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
   spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
   rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
   zcr = librosa.feature.zero_crossing_rate(audio)
   mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate)

   list_of_features.append(np.mean(chroma_stft))
   list_of_features.append(np.mean(rmse))
   list_of_features.append(np.mean(spec_cent))
   list_of_features.append(np.mean(spec_bw))
   list_of_features.append(np.mean(rolloff))
   list_of_features.append(np.mean(zcr))

   for mfcc in mfccs:
         list_of_features.append(np.mean(mfcc))
   
   return(list_of_features)

def calculate_delta(array):

   rows,cols = array.shape
   print(rows)
   print(cols)
   deltas = np.zeros((rows,20))
   Number = 2
   for i in range(rows):
      index = []
      j = 1
      while j <= Number:
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

   sample_rate,audio = read('audio/audio.wav')
   features   = extract_features(audio,sample_rate)

   gmm_files    = ['gmm_models/dina.gmm','gmm_models/kareman.gmm','gmm_models/mariam.gmm','gmm_models/nada.gmm']
   models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
   log_likelihood = np.zeros(len(models)) 
   threshold=-23
   for i in range(len(models)):
      gmm    = models[i]  #checking with each model one by one
      scores = np.array(gmm.score(features))
      log_likelihood[i] = scores.sum()
   print(log_likelihood)
   winner = np.argmax(log_likelihood)
   speakers=['Dina','Kareman','Mariam','Nada']
   positive_scores=[]
   for score in log_likelihood:
      positive_scores.append(-1*score)
   bar=bar_plot(positive_scores,speakers,-1*threshold)
   bar='static/assets/images/bar'+str(variables.counter)+'.jpg'

   if max(log_likelihood)<threshold:
      return 'others',bar
   
   return speakers[winner],bar


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

   speaker,bar=speaker_model()

   print(f'speaker{speaker},words{words}')



   spectrum= visualize("audio/audio.wav")
   spectrum='static/assets/images/result'+str(variables.counter)+'.jpg'
   variables.counter+=1
   spec_fig=spectral_features("audio/audio.wav")
   spec_fig='static/assets/images/spec'+str(variables.counter)+'.jpg'
   mfcc_fig=Mfcc("audio/audio.wav")
   mfcc_fig='static/assets/images/mfcc'+str(variables.counter)+'.jpg'

   
   if speaker=='others':
      return render_template('index.html',words='Unknown Person',spectrum=spectrum,spec_fig=spec_fig,mfcc_fig=mfcc_fig,bar=bar)
   elif word=='others':
      return render_template('index.html',words=f'Hello {speaker}, you entered the wrong password.',spectrum=spectrum,spec_fig=spec_fig,mfcc_fig=mfcc_fig,bar=bar)
   else:
      return render_template('index.html',words=f'Welcome {speaker}!',spectrum=spectrum,spec_fig=spec_fig,mfcc_fig=mfcc_fig,bar=bar)

   

if __name__ == '__main__':
   app.run(debug=True)