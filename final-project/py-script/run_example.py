import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
import IPython
import IPython.display as ipd
matplotlib.style.use('seaborn')

def run_example(wavfile):
    file_name = 'data_overview/audio-sample/{}'.format(wavfile)
    print('Cowbell Labeled Audiofile')
    IPython.display.display(ipd.Audio(file_name))
    x, sr = librosa.load(file_name)
    plt.figure(figsize=(12,3))
    plt.title('Audio Frequency Waveplot')
    librosa.display.waveplot(x, sr=sr)
    X=librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(11.6,3))
    plt.title('Audio File Spectrogram')
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis = 'hz')
    
run_example('0ade0819.wav')