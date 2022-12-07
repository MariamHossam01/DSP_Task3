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

def load_speech_model(x):
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


def extract_features_sound():
   try:
      X, sample_rate = librosa.load('audio.wav', mono=True, duration=30) 
      
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

   except:
      pass

   return ([mfccs, chroma, mel, contrast, tonnetz])


def feat(features_label):
   features = []
   for i in range(0, len(features_label)):
      features.append(np.concatenate((features_label[i][0], features_label[i][1], 
               features_label[i][2], features_label[i][3],
               features_label[i][4]), axis=0))
   return features   

def extarct_features():
   list_of_features=[]
   
   X, sample_rate = librosa.load('audio.wav', mono=True, duration=30)
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

   list_of_features.append(mfccs)
   list_of_features.append(chroma)
   list_of_features.append(mel)
   list_of_features.append(contrast)
   list_of_features.append(tonnetz)
   

   features = []
   for i in range(0, len(list_of_features)):
      features.append(np.concatenate((list_of_features[i][0], list_of_features[i][1], 
               list_of_features[i][2], list_of_features[i][3],
               list_of_features[i][4]), axis=0))
   return features


def load_sound_model(x):
   model = pickle.load(open("trained_sound_model.sav",'rb')) 
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
   sound_features=[]
   # sound_features.append(extarct_features())
   sound_features.append(extract_features_sound())
   # soundFeatures=feat(sound_features)
   persons=load_sound_model(sound_features)
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