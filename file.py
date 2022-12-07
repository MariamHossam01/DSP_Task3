from flask import Flask, render_template, request
import numpy as np
import pickle
import wave
import os
import librosa
import pandas as pd
import speech_recognition as sr
model = pickle.load(open('Classifier.pickle', 'rb'))

def recognize_speech_from_mic(recognizer, microphone):
   """Transcribe speech from recorded from `microphone`.
   Returns a dictionary with three keys:
   "success": a boolean indicating whether or not the API request was
               successful
   "error":   `None` if no error occured, otherwise a string containing
               an error message if the API could not be reached or
               speech was unrecognizable
   "transcription": `None` if speech could not be transcribed,
               otherwise a string containing the transcribed text
   """
   # check that recognizer and microphone arguments are appropriate type
   if not isinstance(recognizer, sr.Recognizer):
      raise TypeError("`recognizer` must be `Recognizer` instance")

   if not isinstance(microphone, sr.Microphone):
      raise TypeError("`microphone` must be `Microphone` instance")

   # adjust the recognizer sensitivity to ambient noise and record audio
   # from the microphone
   with microphone as source:
      recognizer.adjust_for_ambient_noise(source)
      audio = recognizer.listen(source)
      with open('Sound recordings/test/test.wav','wb') as f:
         f.write(audio.get_wav_data())

   # set up the response object
   response = {
      "success": True,
      "error": None,
      "transcription": None
   }

   # try recognizing the speech in the recording
   # if a RequestError or UnknownValueError exception is caught,
   #     update the response object accordingly
   try:
      response["transcription"] = recognizer.recognize_google(audio)
   except sr.RequestError:
      # API was unreachable or unresponsive
      response["success"] = False
      response["error"] = "API unavailable"
   except sr.UnknownValueError:
      # speech was unintelligible
      response["error"] = "Unable to recognize speech"

   return response



app = Flask(__name__,template_folder="templates")

ButtonPressed = 0
# @app.route('/')
# def hello_name():
#    return render_template('index.html')

    
# @app.route('/', methods=['GET', 'POST'])
# def button():
#    global ButtonPressed
#    if request.method == "POST":
#       ButtonPressed += 1
#       print(ButtonPressed)
#       return render_template("index.html", ButtonPressed = ButtonPressed)
#       # I think you want to increment, that case ButtonPressed will be plus 1.
#    # return render_template("index.html", ButtonPressed = ButtonPressed)

# @app.route('/', methods=['GET', 'POST'])
# def record():
#    # create recognizer and mic instances
#    recognizer = sr.Recognizer()
#    microphone = sr.Microphone()
#    guess = recognize_speech_from_mic(recognizer, microphone)
   
#       # show the user the transcription
#    print("You said: {}".format(guess["transcription"]))

#    return render_template("index.html", words = guess["transcription"])


@app.route('/', methods=['GET', 'POST'])
def record():
    # create recognizer and mic instances
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    guess = recognize_speech_from_mic(recognizer, microphone)
    
        # show the user the transcription
    print("You said: {}".format(guess["transcription"]))
    def extract_features(files,name="allSound"):
        try:
            # Sets the name to be the path to where the file is in my computer
            file_name = os.path.join(os.path.abspath('Sound recordings/{}').format(name)+ ('\\') +str(files['file']))

            # Loads the audio file as a floating point time series and assigns the default sample rate
            # Sample rate is set to 22050 by default
            
            X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
            
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

            # We add also the classes of each file as a label at the end
            label = files.label
        
        except:
            print(files)

        return mfccs, chroma, mel, contrast, tonnetz, label

    def feat(features_label):
            features = []
            for i in range(0, len(features_label)):
                features.append(np.concatenate((features_label[i][0], features_label[i][1], 
                features_label[i][2], features_label[i][3],
                features_label[i][4]), axis=0))
            return features   
    #read them into pandas
    filelist = os.listdir('Sound recordings//test//')
    df_test = pd.DataFrame(filelist)
    df_test['label']=0
    df_test = df_test.rename(columns={0:'file'})
    features_label2 = df_test.apply(extract_features,name="test", axis=1)
    features=feat(features_label2)
    prediction = model.predict(features)
    # if prediction == 0:
    #      prediction = "Other"
    # elif prediction == 1:
    #      prediction = "Bassent"
    # elif prediction == 2:
    #      prediction = "Turky"
    # elif prediction == 3:
    #      prediction = "Mayar"
    # elif prediction == 4:
    #      prediction = "Ereny"
    print(prediction)

    return render_template("index.html", words = guess["transcription"],prediction_text=' The speaker is:{}'.format(prediction))



if __name__ == '_main_':
   app.run(debug=True,port=10000)