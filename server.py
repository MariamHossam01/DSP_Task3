from flask import Flask, render_template, request
import numpy as np
import pickle
import sounddevice as sd
import wavio as wv
import librosa


import speech_recognition as sr



app = Flask(__name__,template_folder="templates")

def record():
   # Sampling frequency
   frequency = 44400
   # Recording duration in seconds
   duration = 2
   # to record audio from
   # sound-device into a Numpy
   recording = sd.rec(int(duration * frequency),samplerate = frequency, channels = 2)
   # Wait for the audio to complete
   sd.wait()
   # using wavio to save the recording in .wav format
   # This will convert the NumPy array to an audio
   # file with the given sampling frequency
   wv.write("audio.wav", recording, frequency, sampwidth=2)

def load(x):
   model = pickle.load(open("trainedModel.sav",'rb')) 
   y=model.predict(np.array(x).reshape(1,-1))[0]
   return y

def extractWavFeatures():
   list_of_features=[]
   y, sr = librosa.load('audio.wav', mono=True, duration=30)
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



@app.route('/', methods=['GET', 'POST'])
def speechRecognation():
   record()
   audio_features=[]
   audio_features.append(extractWavFeatures())
   value=load(audio_features)
   print(value)
   return render_template('index.html',words=value)

if __name__ == '__main__':
   app.run(debug=True)