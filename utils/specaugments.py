import os
import librosa
import numpy as np
import librosa.display
from skimage.io import imsave
import tensorflow_io as tfio
from tensorflow_addons.image import sparse_image_warp
import random
import matplotlib.pyplot as plt
# Samples recording frequency made at 16khz
# Features are extracted using default parameters of melspectrogram:
# window = 'hahn', hop_length = 512, n_fft = 2048
# Hop length refers to the number of samples between successive frames
# n_fft is the length of the sampling window,i.e. the number of samples in each window,i.e. the window size
# by default padding is applied at the edges of the signal
# n_mels is the number by which to divide the frequency spectrum of the input signal
# hahn window usage to reduce spectral leakage(acc. to DSP Background, playlist on yt)


def time_warp(spec, policy='LB'):
    """
    This function utilizes time-warp technique, as mentioned in the SpecAugments paper
    :param spec: Spectrogram to be time-warped
    :param policy: Which parameters to use
    :return: A time-warped spectrogram
    """
    if policy == 'LB':
        W, F, m_F, T, p, m_T = 80, 27, 1, 100, 1.0, 1
    elif policy == 'LD':
        W, F, m_F, T, p, m_T = 80, 27, 2, 100, 1.0, 2
    elif policy == 'SM':
        W, F, m_F, T, p, m_T = 40, 15, 2, 70, 0.2, 2
    elif policy == 'SS':
        W, F, m_F, T, p, m_T = 40, 27, 2, 70, 0.2, 2

    v, tau = spec.shape[0], spec.shape[1]

    horiz_line_thru_ctr = spec[v // 2]

    random_pt = horiz_line_thru_ctr[
        random.randrange(W, tau - W)]  # random point along the horizontal/time axis
    w = np.random.uniform((-W), W)  # distance

    # Source Points
    src_points = [[[v // 2, random_pt]]]

    # Destination Points
    dest_points = [[[v // 2, random_pt + w]]]

    mel_spectrogram, _ = sparse_image_warp(spec, src_points, dest_points, num_boundary_points=2)

    return mel_spectrogram


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


def spec_augment(init_path, sr, flag, mode, savepath, mfcc_num):
    """
        This function preprocesses audio signals and returns either mfcc coefficients or mel-spectrograms
        :param init_path: Original path of audio signals
        :param sr: Sample rate of audio signals, used when reading the audio files.
        :param flag: Whether to return mfcc or mel-spectrograms
        :param mode: Spectrogram Augmentation Technique
        :param savepath: Save path for the created mfcc representations or mel-spectrograms
        :param mfcc_num: Number of mfcc coefficients or mel coefficients
        :return: Null
        """
    count = 0
    samples_path = os.listdir(init_path)
    for sample in samples_path:
        filepath = os.path.join(init_path, sample)
        y, _ = librosa.load(filepath, sr=sr, res_type='kaiser_fast', mono=True)
        if flag == 0:
            mel_signal = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=mfcc_num)
            power_to_db = librosa.power_to_db(abs(mel_signal), ref=np.max)
            if mode == 'time_mask':
                time_masked = tfio.audio.time_mask(power_to_db, param=10)
                time_masked = time_masked.numpy()
                # timemask_mfcc = librosa.feature.mfcc(S=time_masked, sr=sr, n_mfcc=mfcc_num)
                image = scale_minmax(time_masked, 0, 255).astype(np.uint8)
                image = np.flip(image, axis=0)
                image = 255 - image
                imsave(savepath.format(count), image)
                print("time_mask_mfcc Image Saved #: ", count)
            elif mode == 'freq_mask':
                frequency_mask = tfio.audio.freq_mask(power_to_db, param=10)
                frequency_mask = frequency_mask.numpy()
                # freqmask_mfcc = librosa.feature.mfcc(S=frequency_mask, sr=sr, n_mfcc=mfcc_num)
                image = scale_minmax(frequency_mask, 0, 255).astype(np.uint8)
                image = np.flip(image, axis=0)
                image = 255 - image
                imsave(savepath.format(count), image)
                print("freq_mask_mfcc Image Saved #: ", count)
            elif mode == 'normal':
                image = scale_minmax(power_to_db, 0, 255).astype(np.uint8)
                image = np.flip(image, axis=0)
                image = 255 - image
                imsave(savepath.format(count), image)
                print("spec imaged saved #:", count)
            else:
                time_warped = time_warp(power_to_db)
                time_warped = time_warped.numpy()
                image = scale_minmax(time_warped, 0, 255).astype(np.uint8)
                image = np.flip(image, axis=0)
                image = 255 - image
                imsave(savepath.format(count), image)
                print("time warped Image Saved #: ", count)
            count += 1
    return 0


savepath1 = 'E:/Stereo_256/Timewarp/image{}.png'  # Train Save paths
savepath2 = 'E:/Stereo_256/Freqmask/image{}.png'  # Test Save paths
savepath3 = 'E:/Stereo_256/Timemask/image{}.png'  # Test Save paths
savepath4 = 'E:/Stereo_256/Train/image{}.png'
savepath5 = 'E:/Stereo_256/Test/image{}.png'
# File paths Data
path = 'E:/Datasets/Train DataSet/'
test_path = 'E:/Datasets/Test DataSet/'
sr1 = 16000
flag = 0
n_mels = 256

spec_augment(path, sr1, flag, 'freq_mask', savepath2, n_mels)
spec_augment(path, sr1, flag, 'time_mask', savepath3, n_mels)
spec_augment(path, sr1, flag, 'time_warp', savepath1, n_mels)

print("End of script")
# EOF
