# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 16:32:59 2018

@author: Abhilash
"""
import librosa
import numpy as np
import matplotlib.pyplot as plt

base_path = "F:\Fall18_Sem1\MLSP\Assignments\Assignment2\data\data\\"

x_source = base_path + "x.wav"

Original_Source, sr1 = librosa.load(x_source, sr=None)

N = 1024

#Create a DFT matrix of NxN
def create_DFT_Matrix(N):
    DFT_matrix = np.array([[np.exp(-1j*(2*np.pi*f*n/N)) for f in range(0,N)] for n in range(0,N)])
    return DFT_matrix

def create_Inverse_DFT_Matrix(N):
    DFT_inv_matrix = np.array([[np.exp(1j*(2*np.pi*f*n/N)) for f in range(0,N)] for n in range(0,N)])/N
    return DFT_inv_matrix

#Creating the data matrix by multiplying Hann window to size N and shifting the window each time by N/2
# X is the input matrix and N is the size of the Hann window
def create_DataMatrix_Using_Hann_Window(source, N):
    n = 0
    #Creating a Hann window
    Hann_window = np.hanning(N)
    X_DataMatrix = []
    while n+N <= len(source):
        x_window = source[n:n+N]
        x_vector = np.multiply(x_window, Hann_window)
        X_DataMatrix.append(x_vector)
        n = n + int(N/2)
    X_DataMatrix = np.array(X_DataMatrix)
    return X_DataMatrix.T


def reverse_Overlap_and_Add(X, N):
    X_out = X[:(X.shape[0] - int(N/2)),0]
    for col in range(1,X.shape[1]):
        X_col = X[(X.shape[0] - int(N/2)):,col-1] + X[0:int(N/2),col]
        X_out = np.concatenate((X_out,X_col))
    
    X_out = X_out.reshape(-1)
    
    return X_out

#DFT matrix
DFT_matrix = create_DFT_Matrix(N)

#Hann windowed X data matrix
X_DataMatrix = create_DataMatrix_Using_Hann_Window(Original_Source, N)
#Y = FX, the spectogram
Y_spectogram = np.dot(DFT_matrix, X_DataMatrix)
Y_spectogram_magnitude = np.abs(Y_spectogram)

#Lets take the last 124 column vectors as noise
Noise_Window = 20
Noise_Model = Y_spectogram_magnitude[:,Y_spectogram_magnitude.shape[1] - Noise_Window:]
Mean_Noise_Model = np.mean(Noise_Model, axis=1).reshape(-1,1)
Y_cleaned_mag = Y_spectogram_magnitude - Mean_Noise_Model

Y_cleaned_mag[Y_cleaned_mag < 0] = 0

Y_original_phase = np.multiply(Y_spectogram, (1/Y_spectogram_magnitude))

Y_cleaned_spectogram = np.multiply(Y_original_phase, Y_cleaned_mag)

Inverse_DFT_matrix = create_Inverse_DFT_Matrix(N)

Fstar_F = np.dot(Inverse_DFT_matrix, DFT_matrix)

X_recovered = np.dot(Inverse_DFT_matrix, Y_cleaned_spectogram)

X_recovered_real = X_recovered.real
X_recovered_time_signal = reverse_Overlap_and_Add(X_recovered_real, N)

output_sound = base_path + 'output.wav'
librosa.output.write_wav(output_sound,X_recovered_time_signal, sr = sr1)

#Plot start
plt.figure(figsize= (5,4))
plt.title("Spectrogram")
plt.imshow(Y_spectogram_magnitude, cmap='jet', aspect='auto')                               #Spectogram with complex values, ploting magnitudes
plt.show()

plt.figure(figsize= (5,4))
plt.title("Spectrogram with exaggerated noise")
plt.imshow(np.power(Y_spectogram_magnitude, float(1/2)), cmap='jet', aspect='auto')         #exaggerated noise Y spectogram
plt.show()

plt.figure(figsize= (5,4))
plt.title("Cleaned Spectrogram")
plt.imshow(np.abs(Y_cleaned_spectogram), cmap='jet', aspect='auto')                         #Cleaned Spectogram magnitudes
plt.show()

plt.figure(figsize= (5,4))
plt.title("Cleaned Spectrogram with Exaggerated noise")
plt.imshow(np.power(np.abs(Y_cleaned_spectogram), float(1/2)), cmap='jet', aspect='auto')   #exaggerated noise Y spectogram
plt.show()

#DFT matrix plot
plt.figure(figsize= (5,4))
plt.title("DFT Matrix real")
plt.imshow(DFT_matrix.real, cmap='jet', aspect='auto')
plt.show()

#DFT matrix plot
plt.figure(figsize= (5,4))
plt.title("DFT Matrix Imaginary")
plt.imshow(DFT_matrix.imag, cmap='jet', aspect='auto')
plt.show()

#Ploting original signal
x_range = range(Original_Source.shape[0])
plt.figure(figsize= (6,3))
plt.title("Original Source signal")
plt.plot(x_range, Original_Source)
plt.show()

#Ploting recovered signal
x_range = range(X_recovered_time_signal.shape[0])
plt.figure(figsize= (6,3))
plt.title("Recovered Time signal")
plt.plot(x_range, X_recovered_time_signal)
plt.show()
#Plot end