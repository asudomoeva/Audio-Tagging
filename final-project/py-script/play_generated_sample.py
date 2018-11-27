import random
import IPython
import IPython.display as ipd
import librosa
import matplotlib.pyplot as plt
import numpy as np


def play_sample(training_results):
    generated_samples = training_results[0][1].generated_x_samples.transpose()
    single_sample = random.randint(0, generated_samples.shape[0] - 1)
    print("Generated Audio File for: Acoustic Guitar")
    IPython.display.display(ipd.Audio(data=generated_samples[single_sample],
                                      rate=22050))
    plt.figure(figsize=(12, 3))
    plt.title('Audio Frequency Waveplot')
    librosa.display.waveplot(np.array(generated_samples[single_sample]),
                             sr=22050)
    plt.show()
