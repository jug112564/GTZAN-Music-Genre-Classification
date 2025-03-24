import librosa
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load(f'./Data/genres_original/classical/classical.00038.wav')

# sound info(text)
print('y:', y)
print('y shape : ', np.shape(y))
print('Sample Rate (KHz) : ', sr)
print('Check Len of Audio : ', 661794/22050)

#drawing wave graph
audio, _ = librosa.effects.trim(y)
print('Audio File:', audio)
print('Audio File shape:', np.shape(audio))

plt.figure(figsize = (13,6))
librosa.display.waveshow(y = audio, sr = sr, color = '#00008B')
plt.title("Example Sound Waves")
plt.show()

#drawing spectogram
n_fft = 2048
hop_length = 512
stft = np.abs(librosa.stft(audio, n_fft = n_fft, hop_length  = hop_length))
print(np.shape(stft))

decibel = librosa.amplitude_to_db(stft ,ref = np.max)

plt.figure(figsize = (13,6))
librosa.display.specshow(decibel, sr = sr, hop_length = hop_length, x_axis = 'time', y_axis = 'log')
plt.colorbar()
plt.show()

#MFCC
mfcc = librosa.feature.mfcc(y=audio, sr=sr)
plt.figure(figsize= (13,6))
librosa.display.specshow(mfcc, sr=sr, x_axis='time')
plt.colorbar()
plt.show()
