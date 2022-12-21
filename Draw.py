import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io 
import base64 
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
class variables:
   counter=0

def  visualize(file_name):
   fig,ax = plt.subplots(figsize=(6,6))
   ax=sns.set_style(style='darkgrid')
   fig.patch.set_facecolor('#e4e8e8')
   signal, sample_rate = librosa.load(file_name)
   first = signal[:int(sample_rate*15)]
   plt.specgram(first, Fs=sample_rate)
   plt.colorbar()
   plt.xlabel("Time (Sec)" )
   plt.ylabel("Frequency (Hz)")
   plt.title(" Audio Spectrogram")
   img=image(fig,"result")
   return img
   

def spectral_features(audio):
   signal , sample_rate = librosa.load(audio)
   spec_bw = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate)
   S, phase = librosa.magphase(librosa.stft(y=signal))
   centroid = librosa.feature.spectral_centroid(S=S)
   rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate, roll_percent=0.99)
   rolloff_min = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate, roll_percent=0.01)
   fig, ax = plt.subplots()
   times = librosa.times_like(spec_bw)
   ax.semilogy(times, spec_bw[0], label='Spectral bandwidth')
   ax.legend()
   ax.label_outer()
   fig.patch.set_facecolor('#e4e8e8')
   img=librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         y_axis='log', x_axis='time', ax=ax)
   ax.set(title='The Spectral Features ')
   ax.fill_between(times, np.maximum(0, centroid[0] - spec_bw[0]),
                  np.minimum(centroid[0] + spec_bw[0], sample_rate/2),
                  alpha=0.5, label='Centroid +- bandwidth')
   ax.plot(times, centroid[0], label='Spectral centroid', color='w')
   ax.plot(librosa.times_like(rolloff), rolloff[0], label='Roll-off frequency (0.99)')
   ax.plot(librosa.times_like(rolloff), rolloff_min[0], color='w',
         label='Roll-off frequency (0.01)')
   ax.legend(loc='lower right')
   fig.colorbar(img, ax=ax)
   img=image(fig,"spec")
   return img


def Mfcc(audio):
   signal,sample_rate=librosa.load(audio)
   mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40)
   fig, ax = plt.subplots()
   fig.patch.set_facecolor('#e4e8e8')
   img = librosa.display.specshow(mfccs, x_axis='time',y_axis="log",ax=ax)
   fig.colorbar(img, ax=ax)
   ax.set(title=' The MFCC Feature')
   img=image(fig,"mfcc")
   return img

def image(fig,name):
   canvas=FigureCanvas(fig)
   img=io.BytesIO()
   fig.savefig(img, format='png')
   img.seek(0)
   data = base64.b64encode(img.getbuffer()).decode("ascii")
   image_file_name='static/assets/images/'+str(name)+str(variables.counter)+'.jpg'
   plt.savefig(image_file_name)
   return f"<img src='data:image/png;base64,{data}'/>"

def bar_plot(scores,speakers,threshold):
   values = np.array(scores)
   # split it up
   above_threshold = np.maximum(values - threshold, 0)
   below_threshold = np.minimum(values, threshold)

   # and plot it
   fig, ax = plt.subplots()
   fig.patch.set_facecolor('#e4e8e8')
   ax.bar(speakers, below_threshold, 0.5, color="#43448e")
   ax.bar(speakers, above_threshold, 0.5, color="#b30003",
         bottom=below_threshold)
   ax.set(title=' Absolute Likelihood')
   # creating the bar plot
   img=image(fig,"bar")
   return img


