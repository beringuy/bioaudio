import pandas as pd
import numpy as np
import os
import shutil

from pydub import AudioSegment
from BIOAUDIO.bioaudio_util import maxAudioLengthDetect, makeAudiosSameLength, audioCutter, plot_spectrogram, mel_spectrogram

import splitfolders

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import torchaudio

import matplotlib.pyplot as plt

'''
LINHAS COM "# < < < < < < < < < < < < < < < <" PRECISAM SER AJUSTADAS DE ACORDO COM AS ESPECIFICIDADES DO DATASET USADO
'''

# # # # # # # # # # # #
# AJUSTES INICIAS (PASTAS E INFORMAÇÕES)

#mainFolder = "/home/pc/Downloads/BIOAUDIO/"
mainFolder = "/home/pc/Downloads/TESTE/" # < < < < < < < < < < < < < < < < # < < < < < < < < < < < < < < < <

# carregando informações sobre o dataset:
infoFile = mainFolder + "metadata_myVersion.csv" # < < < < < < < < < < < < < < < < # < < < < < < < < < < < < < < < <
df = pd.read_csv(infoFile)

# # # # # # # # # # # #

silenced = mainFolder + "2_silenced/" # pasta que armazenará os arquivos após o processo de remoção do silêncio
sameLength = mainFolder + "3_sameLength/" # nem sempre usado / adicionar trechos do código se necessário
spectro = mainFolder + "4_spectro/"
sortedByClass = mainFolder + "5_sortedByClass/"
forDL = mainFolder + "6_forDL/"

os.makedirs(os.path.dirname(silenced), exist_ok=True)
os.makedirs(os.path.dirname(sameLength), exist_ok=True)
os.makedirs(os.path.dirname(spectro), exist_ok=True)
os.makedirs(os.path.dirname(sortedByClass), exist_ok=True)
os.makedirs(os.path.dirname(forDL), exist_ok=True)

# # # # # # # # # # # #
# REMOÇÃO DE SILÊNCIO:

toSilence = mainFolder + "samples/" # pasta com TODOS os arquivos .wav a serem utilizados # < < < < < < < < < < < < < < < < # < < < < < < < < < < < < < < < <

tmpExcepts = 0

for item in os.listdir(toSilence):
    name, ext = os.path.splitext(item)
    print("\nname: ", name, "& ext: ", ext, "\n")
    if ext == '.wav':
        try:
            tmp, tmpLength = audioCutter(item, toSilence, silenced) # remove os trechos considerados sileciosos, ajustar conforme necessidade
        except:
            print ("Exception: " + name + ext)
            tmpExcepts += 1

print("\nExceptions:", tmpExcepts)


# # # # # # # # # # # #
# ESPECTROGRAMAS:

originToSpectro = silenced # extraindo os espectrogramas a partir dos arquivos após a remoção do silêncio

for item in os.listdir(originToSpectro):
    name, ext = os.path.splitext(item)
    print("\nname: ", name, "& ext: ", ext, "\n")
    if ext == '.wav':
        tmp_data_waveform, tmp_rate_of_sample = torchaudio.load(originToSpectro + name + ext)        
        melspec = mel_spectrogram(tmp_data_waveform)
        plot_spectrogram(melspec[0], name + ext, originToSpectro) # extrai para a mesma pasta dos audios usados


# move os spectrogramas para a pasta correta:    
for item in os.listdir(originToSpectro):
    name, ext = os.path.splitext(item)
    print("\nname: ", name, "& ext: ", ext, "\n")
    if ext == '.png':
        shutil.copy(originToSpectro + name + ext, spectro) # .move / .copy


# # # # # # # # # # # #
# CLASSES:

# muda o identificador único do arquivo e a coluna que define a classe do arquivo:
df.columns
df.rename(columns={'uuid': 'file_name', 'status': 'class'}, inplace=True) # < < < < < < < < < < < < < < < < # < < < < < < < < < < < < < < < <


originToOrganize = spectro
# função que move o espectrograma para determinada pasta, com base na classe do arquivo:
def classMoover (classTag, originToOrganize, tmpDestiny):
    for file, format in zip(df[df["class"] == classTag]["file_name"], df[df["class"] == classTag]["fileFormat"]):
        tmpFileToCopyPath = originToOrganize + file + format + ".wav.png" ### <<< ! <<< formato atual adaptado para nomenclatura específica # < < < < < < < < < < < < < < < <
        try:
            shutil.copy(tmpFileToCopyPath, tmpDestiny)
        except:
            pass


# classes presentes no Dataset:
df["class"].unique()

# seleção dos arquivos com base nas classes de interesse:
df = df.loc[df['class'].isin(["healthy", "COVID-19"])] # < < < < < < < < < < < < < < < < # < < < < < < < < < < < < < < < <

classList = df["class"].unique()

# criando pastas para cada classe e movendo/copiando o espectrograma para a pasta criada, de acordo com a classe do arquivo
for class_ in classList:    
    tmpDestiny = sortedByClass + class_ + "/"
    os.makedirs(os.path.dirname(tmpDestiny), exist_ok=True)
    classMoover(class_, originToOrganize, tmpDestiny)

# # # # # # # # # # # #
# TREINO / TESTE / VALIDAÇÃO:

splitfolders.ratio(sortedByClass,
                   output = forDL,
                   ratio = (.8, .1, .1))


#+++++++++++++++++++++++++# #+++++++++++++++++++++++++# #+++++++++++++++++++++++++#

# # # # # # # # # # # #
# CONVOLUTIONAL NEURAL NETWORK:

model = Sequential()

model.add(Conv2D(8, (3,3), input_shape = (496, 369, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2), padding = "same"))

model.add(Conv2D(8, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(8, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(8, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))


model.add(Flatten())
model.add(Dense(64)) # original 128
model.add(Activation("relu"))
model.add(Dense(1))
model.add(Activation("sigmoid"))

#model.compile(loss = "binary_crossentropy", optimizer="adam", metrics = ["accuracy"])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.FalseNegatives()])

              
model.summary()

batch_size = 100


# # # # # # # # # # # #
# PREPARANDO OS DADOS:

# # # # # # # # # # # # # # #
# Estrutura das pastas:
# folder > train > classe 1
#                > classe 2
#        > test  > classe 1
#                > classe 2
#
# # # # # # # # # # # # # # #

trainPath = forDL + 'train/'
valPath = forDL + 'val/'
testPath = forDL + 'test/'

datagen = ImageDataGenerator(rescale = None,  # original = 1./255
                                   #shear_range = 0.2, # original = 0.2
                                   #zoom_range = 0.2, # original = 0.2                                   
                                   horizontal_flip = False) # original = True

# Generates batches of Augmented Image data
train_generator = datagen.flow_from_directory(trainPath,
                                                    target_size = (496, 369),
                                                    batch_size = batch_size,
                                                    color_mode = "grayscale",
                                                    class_mode = 'binary') 

# Generator for validation data
validation_generator = datagen.flow_from_directory(valPath,
                                                        target_size = (496, 369),
                                                        batch_size = batch_size,
                                                        color_mode = "grayscale",
                                                        class_mode = 'binary')

# Generator for test data
test_generator = datagen.flow_from_directory(testPath,
                                                        target_size = (496, 369),
                                                        batch_size = batch_size,
                                                        color_mode = "grayscale",
                                                        class_mode = 'binary')


# # # # # # # # # # # #
# TREINO E AVALIAÇÃO:
# Fit the model on Training data
model.fit(train_generator, 
                    epochs = 12, # 8~12
                    validation_data = validation_generator,
                    verbose = 1)

# Evaluating model performance on Testing data
model.evaluate(test_generator)
model.evaluate(validation_generator)
#loss, accuracy = model.evaluate(validation_generator)


# # # # # # # # # # # # # # # # # # # # # # # #
# SALVA O MODELO GERADO
# CUIDADO PARA NÃO SALVAR POR CIMA DE UM BOM MODELO JÁ EXISTENTE
model.save(mainFolder + "Model/")
# # # # # # # # # # # # # # # # # # # # # # # #