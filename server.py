import os
from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
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
   wv.write("audio/audio.wav", recording, frequency, sampwidth=2)

def load_speech_model(x):
   model = pickle.load(open("trainedModel.sav",'rb')) 
   y=model.predict(np.array(x).reshape(1,-1))[0]
   return y

def extractWavFeatures():
   list_of_features=[]
   y, sr = librosa.load('audio/audio.wav', mono=True, duration=30)
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

def extractSpeakerFeatures():
   list_of_features=[]
   X, sample_rate = librosa.load('audio/audio.wav', mono=True, duration=30)
   
   # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series 
   mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

   # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
   stft = np.abs(librosa.stft(X))

   # Computes a chromagram from a waveform or power spectrogram.
   chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

   # Computes a mel-scaled spectrogram.
   mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

   # Computes spectral contrast
   contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

   # Computes the tonal centroid features (tonnetz)
   tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
   sr=sample_rate).T,axis=0)

   for e in chroma:
      list_of_features.append (np.mean(e))
   for e in mel:
      list_of_features.append (np.mean(e))
   for e in contrast:
      list_of_features.append (np.mean(e))
   for e in tonnetz:
      list_of_features.append (np.mean(e))
   for e in mfccs:
      list_of_features.append (np.mean(e))
   
   return(list_of_features)


def load_sound_model(x):
   model = pickle.load(open("trained_speaker_model.sav",'rb')) 
   y=model.predict(np.array(x).reshape(1,-1))[0]
   return y


@app.route('/', methods=['GET', 'POST'])
def speechRecognation():
   # words
   record()

   speech_features=[]
   speech_features.append(extractWavFeatures())
   words=load_speech_model(speech_features)
   if words==0:
      word='open the door'
   else:
      word='close the door'
   # speaker
   speaker_features=[]
   speaker_features.append(extractSpeakerFeatures())
   persons=load_sound_model(speaker_features)
   if persons==0:
      person='Dina'
   elif persons==1:
      person='Kareman'
   elif persons==2:
      person='Mariam'
   elif persons==3:
      person='Nada'
   else:
      person='others'

   
   return render_template('index.html',words=word,persons=person)

if __name__ == '__main__':
   app.run(debug=True)