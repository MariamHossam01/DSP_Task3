from flask import Flask, render_template, request
import numpy as np
import pickle


import speech_recognition as sr


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

@app.route('/', methods=['GET', 'POST'])
def record():
   # create recognizer and mic instances
   recognizer = sr.Recognizer()
   microphone = sr.Microphone()
   guess = recognize_speech_from_mic(recognizer, microphone)
      # show the user the transcription
   print("You said: {}".format(guess["transcription"]))
   return render_template("index.html", words = guess["transcription"])

if __name__ == '__main__':
   app.run(debug=True)