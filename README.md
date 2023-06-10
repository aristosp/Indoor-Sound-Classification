# Indoor-Sound-Classification
This project was my thesis. Indoor Sound Classification consists of the classification of 9 different home activities, such as using the vacuum cleaner or eating.
An extensive research was conducted, resulting in many different models, beginning with 1D Convolutional Models, 2D Convolutional Models, an ensemble and finally a Vision Transformer Model.
The dataset used is the DCASE2018 Task 5 (https://dcase.community/challenge2018/task-monitoring-domestic-activities).

Due to the high computational power needed to classify the original audio files, the sounds are converted to MFCC or spectrograms. Before the conversion is made, preprocessing the audio files is necessary.

![githubimage1](https://github.com/aristosp/Indoor-Sound-Classification/assets/62808962/b062d0b9-48ac-4d26-b347-4cf2c66dc86b)

Data augmenantion was used, consinsting of those presented in SpecAugment (https://arxiv.org/abs/1904.08779), resulting in the following augmentations:
![image](https://github.com/aristosp/Indoor-Sound-Classification/assets/62808962/521180e5-37df-4d43-a2ba-f4ce5ba898f0)


