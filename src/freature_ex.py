import os
import numpy as np
import scipy
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import get_window
import librosa



def normalize_audio(audio):
         audio = audio / np.max(np.abs(audio))
         return audio


def frame_audio(audio, FFT_size=2048, hop_size=10, sample_rate=44100):
         # hop_size in ms
         
         audio = np.pad(audio, int(FFT_size / 2), mode='reflect')
         frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
         frame_num = int((len(audio) - FFT_size) / frame_len) + 1
         frames = np.zeros((frame_num,FFT_size))
         
         for n in range(frame_num):
             frames[n] = audio[n*frame_len:n*frame_len+FFT_size]
         
         return frames

def freq_to_mel(freq):
         return 2595.0 * np.log10(1.0 + freq / 700.0)
     
def met_to_freq(mels):
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)

def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=44100):
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
    freqs = met_to_freq(mels)
    
    return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs     


def get_filters(filter_points, FFT_size):
         filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
         
         for n in range(len(filter_points)-2):
             filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
             filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
         
         return filters

def dct(dct_filter_num, filter_len):
         basis = np.empty((dct_filter_num,filter_len))
         basis[0, :] = 1.0 / np.sqrt(filter_len)
         
         samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)
     
         for i in range(1, dct_filter_num):
             basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
             
         return basis

# def _spectrogram(stft=None,hop_length=512,power=1,win_length=None, window="hann",center=True,pad_mode="constant",):
#            if FFT_size // 2 + 1 != stft.shape[-2]:
#                FFT_size = 2 * (stft.shape[-2] - 1)
#            return stft, FFT_size

def get_mfcc(audio,sample_rate):
     
     
     audio = normalize_audio(audio)
     hop_size = 15 #ms
     FFT_size = 2048
     
     audio_framed = frame_audio(audio, FFT_size=FFT_size, hop_size=hop_size, sample_rate=sample_rate)
     window = get_window("hann", FFT_size, fftbins=True)
     audio_win = audio_framed * window
     
     audio_winT = np.transpose(audio_win)
     
     audio_fft = np.empty((int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')
     
     for n in range(audio_fft.shape[1]):
         audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]
     
     audio_fft = np.transpose(audio_fft)
     audio_power = np.square(np.abs(audio_fft))
     freq_min = 0
     freq_high = sample_rate / 2
     mel_filter_num = 10
     filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=44100)
     
     filters = get_filters(filter_points, FFT_size)
     enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
     filters *= enorm[:, np.newaxis]
     audio_filtered = np.dot(filters, np.transpose(audio_power))
     audio_log = 10.0 * np.log10(audio_filtered)
     
     dct_filter_num = 40
     
     dct_filters = dct(dct_filter_num, mel_filter_num)
     
     cepstral_coefficents = np.dot(dct_filters, audio_log)
     return cepstral_coefficents[:, 0]

def get_stft(signal, sample_rate):
     frame_size = 0.050  # with a frame size of 50 milliseconds
     hop_size = 0.025         # and hop size of 25 milliseconds.
     frame_samp = int(frame_size*sample_rate)
     hop_samp = int(hop_size*sample_rate)
     Hanning_window = np.hanning(frame_samp)
     X = np.array([np.fft.fft(Hanning_window*signal[i:i+frame_samp])
                    for i in range(0, len(signal)-frame_samp, hop_samp)])
     return X

def get_mel_spectrogram(signal,sample_rate):  
    FFT_size = 2048
    hop_length=512
    n_chroma=12

    if FFT_size // 2 + 1 != signal.shape[-2]:
        FFT_size = 2 * (signal.shape[-2] - 1)
    freq_min = 0
    freq_high = sample_rate / 2
    mel_filter_num = 128
    filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=44100)
    fftfreqs=np.fft.rfftfreq(n=FFT_size, d=1.0/sample_rate)
    fdiff = np.diff(mel_freqs)
    ramps = np.subtract.outer(mel_freqs, fftfreqs)
    weights = np.zeros((mel_filter_num, int(1 + FFT_size // 2)))

    for i in range(mel_filter_num):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
     # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    enorm = 2.0 / (mel_freqs[2 : mel_filter_num + 2] - mel_freqs[:mel_filter_num])
    weights *= enorm[:, np.newaxis]
    mel_basis=weights
    return np.einsum("...ft,mf->...mt", signal, mel_basis, optimize=True)

# file_name=r"D:\carrie works\dsp_task3\dsp_task3\dsp_task3\index.wav"
# X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
# stft = np.abs(get_stft(X,sample_rate))
# chroma = np.mean(get_chroma_stft(stft,sample_rate).T,axis=0)
# print(stft)
# print(chroma)