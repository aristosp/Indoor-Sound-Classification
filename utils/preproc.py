import os
import librosa
import numpy as np
import librosa.display
from skimage.io import imsave
# This script supposes that files are saved into a local path.

# Samples' recording frequency is 16khz
# Features are extracted using default parameters of melspectrogram:
# window = 'hahn', hop_length = 512, n_fft = 2048
# Hop length refers to the number of samples between successive frames
# n_fft is the length of the sampling window,i.e. the number of samples in each window,i.e. the window size
# by default padding is applied at the edges of the signal
# n_mels is the number by which to divide the frequency spectrum of the input signal
# hahn window usage to reduce spectral leakage(acc. to DSP Background, playlist on yt)


def scale_minmax(X, minimum, maximum):
    """
    A simple function used to rescale its input array to a range from [minimum, maximum]
    :param X: Array to be rescaled
    :param minimum: Minimum range
    :param maximum: Maximim range
    :return: A rescaled array
    """
    x_std = (X - X.min()) / (X.max() - X.min())
    x_scaled = x_std * (maximum - minimum)
    return x_scaled


def preproc_samples(init_path, sr, flag, dataset, savepath, mfcc_num):
    """
    This function preprocesses audio signals and returns either mfcc coefficients or mel-spectrograms
    :param init_path: Original path of audio signals
    :param sr: Sample rate of audio signals, used when reading the audio files.
    :param flag: Whether to return mfcc or mel-spectrograms
    :param dataset: Whether the dataset used is train or test set
    :param savepath: Save path for the created mfcc representations or mel-spectrograms
    :param mfcc_num: Number of mfcc coefficients or mel coefficients
    :return: Null
    """
    count = 0
    samples_path = os.listdir(init_path)
    for sample in samples_path:
        filepath = os.path.join(init_path, sample)
        y, _ = librosa.load(filepath, sr=sr, res_type='kaiser_fast', mono=False)
        if flag == 0:
            mel_signal = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=mfcc_num)
            for channel in range(len(mel_signal)):
                power_to_db = librosa.power_to_db(abs(mel_signal[channel]), ref=np.max)
                image = scale_minmax(power_to_db, 0, 255).astype(np.uint8)
                image = np.flip(image, axis=0)
                image = 255 - image
                if dataset == 'train':
                    imsave(savepath.format(count), image)
                    print("Train Image Saved #:", count)
                    count += 1
                else:
                    imsave(savepath.format(count), image)
                    print("Test Image Saved #:", count)
        elif flag == 1:
            for channel in range(len(y)):
                mfcc = librosa.feature.mfcc(y=y[channel], sr=sr, n_mfcc=mfcc_num)
                image = scale_minmax(mfcc, 0, 255).astype(np.uint8)
                image = np.flip(image, axis=0)
                image = 255 - image
                if dataset == 'train':
                    imsave(savepath.format(count), image)
                    print("Train Image Saved #:", count, " for channel", channel)
                    count += 1
                else:
                    imsave(savepath.format(count), image)
                    print("Test Image Saved #:", count, " for channel", channel)
                    count += 1
    return 0


# Train and test images save paths for mel-spectrograms
savepath1 = ['E:/Stereo/Spec/32/Train/spec{}.png',
             'E:/Stereo/Spec/64/Train/spec{}.png',
             'E:/Stereo/Spec/128/Train/spec{}.png']
savepath2 = ['E:/Stereo/Spec/32/Test/spec{}.png',
             'E:/Stereo/Spec/64/Test/spec{}.png',
             'E:/Stereo/Spec/128/Test/spec{}.png']
# Audio files save paths
path = 'E:/Datasets/Train DataSet/'
test_path = 'E:/Datasets/Test DataSet/'
sr = 16000
# Set spectrogram mode
flag = 0
n_mels = [32, 64, 128]
for i in range(len(n_mels)):
    # preproc_samples(path, sr, flag, 'train', savepath1, n_mels[i])
    # Test Data
    preproc_samples(test_path, sr, flag, 'test', savepath2,  n_mels[i])

# Save paths for train and test mfcc image representations
savepath3 = ['E:/Stereo/MFCC/32/Train/mfcc{}.png',
             'E:/Stereo/MFCC/64/Train/mfcc{}.png',
             'E:/Stereo/MFCC/128/Train/mfcc{}.png']  # Train Save paths

savepath4 = ['E:/Stereo/MFCC/32/Test/mfcc{}.png',
             'E:/Stereo/MFCC/64/Test/mfcc{}.png',
             'E:/Stereo/MFCC/128/Test/mfcc{}.png']  # Test Save paths
# Change mode to MFCC
flag = 1
mfccnum = [32, 64, 128]
for i in range(0, len(mfccnum)):
    preproc_samples(path, sr, flag, 'train', savepath3, mfccnum[i])
    # Test Data
    preproc_samples(test_path, sr, flag, 'test', savepath4, mfccnum[i])

print("End of script")
# EOF
