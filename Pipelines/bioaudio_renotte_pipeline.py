
# Build a Deep Audio Classifier with Python and Tensorflow, by Nicholas Renotte
# https://youtu.be/ZLIPkmmDJAc

import os
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf 
import tensorflow_io as tfio

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout

from itertools import groupby

import csv

###########################################################

'''
coughFileList = os.listdir('/home/ego/Datasets/experimento_soloCough/')

df = pd.DataFrame(coughFileList, columns =['file_name'])
df['class'] = 'cough'
df
'''

###########################################################

# # # # # # # # # #
# Build Data Loading Function
## Define Paths to Files
SOURCE = '/home/alessandro/Alessandro/coughDatabaseExp'

## Build Dataloading Function
def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels) 
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

'''
## Plot Wave
wave = load_wav_16k_mono(CAPUCHIN_FILE)
nwave = load_wav_16k_mono(NOT_CAPUCHIN_FILE)
plt.plot(wave)
plt.plot(nwave)
plt.show()
'''

# # # # # # # # # #
# Create Tensorflow Dataset
## Define Paths to Positive and Negative Data
POS = os.path.join(SOURCE, 'soloCough')
NEG = os.path.join(SOURCE, 'noCough')

## Create Tensorflow Datasets
pos = tf.data.Dataset.list_files(POS+'/*.wav')
neg = tf.data.Dataset.list_files(NEG+'/*.wav')

## Add labels and Combine Positive and Negative Samples
positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
data = positives.concatenate(negatives)

# # # # # # # # # #
# Determine Average Length of a Capuchin Call
## Calculate Wave Cycle Length
lengths = []
for file in os.listdir(os.path.join(SOURCE, 'soloCough')):
    tensor_wave = load_wav_16k_mono(os.path.join(SOURCE, 'soloCough', file))
    lengths.append(len(tensor_wave))
    
## Calculate Mean, Min and Max
tf.math.reduce_mean(lengths)
tf.math.reduce_min(lengths)
tf.math.reduce_max(lengths)

# # # # # # # # # #
# Build Preprocessing Function to Convert to Spectrogram
## Build Preprocessing Function
def preprocess(file_path, label): 
    wav = load_wav_16k_mono(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label

## Test Out the Function and Viz the Spectrogram
filepath, label = positives.shuffle(buffer_size=10000).as_numpy_iterator().next()

spectrogram, label = preprocess(filepath, label)

'''
plt.figure(figsize=(30,20))
plt.imshow(tf.transpose(spectrogram)[0])
plt.show()
'''

# # # # # # # # # #
# Create Training and Testing Partitions
## Create a Tensorflow Data Pipeline
data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)

## Split into Training and Testing Partitions
train = data.take(36)
test = data.skip(36).take(15)

## Test One Batch
samples, labels = train.as_numpy_iterator().next()
samples.shape

#####################################

# # # # # # # # # #
# Build Deep Learning Model
## Build Sequential Model, Compile and View Summary
model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(1491, 257,1)))
#model.add(Dropout(0.5)) # <<<
model.add(Conv2D(16, (3,3), activation='relu'))
#model.add(Dropout(0.5)) # <<<
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

model.summary()

#####################################
## Fit Model, View Loss and KPI Plots
hist = model.fit(train, epochs=4, validation_data=test) # < < < < <
'''
plt.title('Loss')
plt.plot(hist.history['loss'], 'r')
plt.plot(hist.history['val_loss'], 'b')
plt.show()

plt.title('Precision')
plt.plot(hist.history['precision'], 'r')
plt.plot(hist.history['val_precision'], 'b')
plt.show()

plt.title('Recall')
plt.plot(hist.history['recall'], 'r')
plt.plot(hist.history['val_recall'], 'b')
plt.show()
'''
#####################################

# # # # # # # # # #
# Make a Prediction on a Single Clip
## Get One Batch and Make a Prediction
X_test, y_test = test.as_numpy_iterator().next()

yhat = model.predict(X_test)
yhat

## Convert Logits to Classes
yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]
yhat

# # # # # # # # # #
# Build Forest Parsing Functions
## Load up MP3s
def load_mp3_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    res = tfio.audio.AudioIOTensor(filename)
    # Convert to tensor and combine channels 
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2 
    # Extract sample rate and cast
    sample_rate = res.rate
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Resample to 16 kHz
    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
    return wav

# # #
#mp3 = os.path.join(CAPUCHIN_SOURCE, 'Forest Recordings', 'recording_00.mp3') # <<<<<<<<<<<<<<<<<<<<<<<<<<<< arquivo para detectar o padrão
mp3 = '/home/alessandro/Alessandro/coughDatabaseExp/FIS_ambient_toDetect/fis06  .mp3'

wav = load_mp3_16k_mono(mp3)

audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1)

samples, index = audio_slices.as_numpy_iterator().next()

## Build Function to Convert Clips into Windowed Spectrograms
def preprocess_mp3(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

## Convert Longer Clips into Windows and Make Predictions
audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=16000, sequence_stride=16000, batch_size=1)
audio_slices = audio_slices.map(preprocess_mp3)
audio_slices = audio_slices.batch(64)

yhat = model.predict(audio_slices)
yhat
yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]
yhat

## Group Consecutive Detections
yhat = [key for key, group in groupby(yhat)]
yhat
calls = tf.math.reduce_sum(yhat).numpy()

calls

# # # # # # # # # #
# Make Predictions
## Loop over all recordings and make predictions
results = {}
for file in os.listdir(os.path.join(SOURCE, 'FIS_ambient_toDetect')):
    FILEPATH = os.path.join(SOURCE,'FIS_ambient_toDetect', file)
    
    wav = load_mp3_16k_mono(FILEPATH)
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1)
    audio_slices = audio_slices.map(preprocess_mp3)
    audio_slices = audio_slices.batch(64)
    
    yhat = model.predict(audio_slices)
    
    results[file] = yhat
    
results

## Convert Predictions into Classes
class_preds = {}
for file, logits in results.items():
    class_preds[file] = [1 if prediction > 0.99 else 0 for prediction in logits]
class_preds

## Group Consecutive Detections
postprocessed = {}
for file, scores in class_preds.items():
    postprocessed[file] = tf.math.reduce_sum([key for key, group in groupby(scores)]).numpy()
postprocessed

'''
# # # # # # # # # #
# Export Results
with open('results.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['recording', 'capuchin_calls'])
    for key, value in postprocessed.items():
        writer.writerow([key, value])
'''